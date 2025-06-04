import os
import logging
from argparse import Namespace
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from transformer_lens.HookedTransformer import Output
from transformer_lens.hook_points import HookPoint
import transformer_lens.utils as utils
from transformer_lens.utilities import devices
from transformer_lens.utils import USE_DEFAULT_VALUE
from transformer_lens.components import AbstractAttention

def gumbel_sigmoid(logits, gs_temp=1., eps=1e-10):
    uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0, 1)
    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / gs_temp)
    res = ((res > 0.5).type_as(res) - res).detach() + res
    return res

def calculate_qkv_matrices(self, query_input, key_input, value_input):

    q = self.hook_q(
        einops.einsum(query_input, self.W_Q,
            "batch query_pos n_heads d_model, n_heads d_model d_head -> batch query_pos n_heads d_head") + self.b_Q
    )
    k = self.hook_k(
        einops.einsum(
            key_input, self.W_K,
            "batch key_pos n_heads d_model, n_heads d_model d_head -> batch key_pos n_heads d_head") + self.b_K
    )
    v = self.hook_v(
        einops.einsum(value_input, self.W_V,
            "batch key_pos n_heads d_model, n_heads d_model d_head -> batch key_pos n_heads d_head") + self.b_V
    )

    return q, k, v

def attn_forward(self, query_input, key_input, value_input,
        past_kv_cache_entry = None,
        additive_attention_mask = None,
        attention_mask = None,
        position_bias = None,
        return_z = False,
    ):

        q, k, v = self.calculate_qkv_matrices(query_input, key_input, value_input)

        if past_kv_cache_entry is not None:
            # Appends the new keys and values to the cached values, and automatically updates the cache
            kv_cache_pos_offset = past_kv_cache_entry.past_keys.size(1)
            k, v = past_kv_cache_entry.append(k, v)
        else:
            # Not using a cache
            kv_cache_pos_offset = 0

        if self.cfg.positional_embedding_type == "rotary":
            q = self.hook_rot_q(self.apply_rotary(q, kv_cache_pos_offset, attention_mask))
            k = self.hook_rot_k(
                self.apply_rotary(k, 0, attention_mask)
            )  # keys are cached so no offset

        if self.cfg.dtype not in [torch.float32, torch.float64]:
            # If using 16 bits, increase the precision to avoid numerical instabilities
            q = q.to(torch.float32)
            k = k.to(torch.float32)

        attn_scores = self.calculate_attention_scores(
            q, k
        )  # [batch, head_index, query_pos, key_pos]

        if self.cfg.positional_embedding_type == "alibi":
            query_ctx = attn_scores.size(-2)
            # The key context length is the number of positions in the past - this includes all positions in the cache
            key_ctx = attn_scores.size(-1)

            # only recompute when necessary to increase efficiency.
            if self.alibi is None or key_ctx > self.alibi.size(-1):
                self.alibi = AbstractAttention.create_alibi_bias(
                    self.cfg.n_heads, key_ctx, self.cfg.device
                )

            attn_scores += self.alibi[
                :, :query_ctx, :key_ctx
            ]  # [batch, head_index, query_pos, key_pos]
        elif self.cfg.positional_embedding_type == "relative_positional_bias":
            if position_bias is None:
                if self.has_relative_attention_bias:
                    raise ValueError("Positional bias is required for relative_positional_bias")
                else:
                    position_bias = torch.zeros(
                        1,
                        self.cfg.n_heads,
                        attn_scores.shape[2],
                        attn_scores.shape[3],
                        device=attn_scores.device,
                    )

            attn_scores += position_bias
        if self.cfg.attention_dir == "causal":
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = self.apply_causal_mask(
                attn_scores, kv_cache_pos_offset, attention_mask
            )  # [batch, head_index, query_pos, key_pos]
        if additive_attention_mask is not None:
            attn_scores += additive_attention_mask

        attn_scores = self.hook_attn_scores(attn_scores)
        pattern = F.softmax(attn_scores, dim=-1)
        pattern = torch.where(torch.isnan(pattern), torch.zeros_like(pattern), pattern)
        pattern = self.hook_pattern(pattern)  # [batch, head_index, query_pos, key_pos]
        pattern = pattern.to(self.cfg.dtype)
        pattern = pattern.to(v.device)
        z = self.calculate_z_scores(v, pattern)  # [batch, pos, head_index, d_head]
        if return_z:
            return z

        attn_out = einops.einsum(z, self.W_O,
            "batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos n_heads d_model") + self.b_O
        return attn_out

class TransformerBlock(nn.Module):
    '''TransformerBlock wrapper'''

    def __init__(self, block, block_index, mask_cfg, mask_control):
        super().__init__()
        self.block = block
        self.block_index = block_index
        self.mask_cfg = mask_cfg
        self.cfg = self.block.cfg
        self.mask_control = mask_control

        self.copy_all_hooks()
        self.copy_components()
        self.device = next(self.parameters()).device
        self.init_masks()

    def copy_hook(self, hook_name):
        hook = getattr(self.block, hook_name, None)
        if hook is None:
            setattr(self, hook_name, HookPoint())
        else:
            setattr(self, hook_name, hook)

    def copy_all_hooks(self):
        
        self.copy_hook('hook_resid_pre')
        self.copy_hook('hook_attn_in')
        self.copy_hook('hook_q_input')
        self.copy_hook('hook_k_input')
        self.copy_hook('hook_v_input')
        self.copy_hook('hook_attn_out')
        self.copy_hook('hook_resid_mid')
        self.copy_hook('hook_resid_post')

    def copy_components(self):

        self.apply_mlp = getattr(self.block, 'apply_mlp', None)
        if self.apply_mlp is None:
            self.apply_mlp = self.mlp

        self.attn = self.block.attn
        self.ln1 = self.block.ln1
        self.ln2 = self.block.ln2

    def init_masks(self):
        pass

    def forward(self, resid_pre, shortformer_pos_embed=None, past_kv_cache_entry=None, attention_mask=None):
        
        resid_pre = self.step_resid_pre(resid_pre)
        attn_in = self.step_attn_in(resid_pre, shortformer_pos_embed)
        query_input, key_input, value_input = self.step_qkv_input(resid_pre, attn_in)
        attn_out = self.step_attn_out(
            query_input, key_input, value_input,
            shortformer_pos_embed=shortformer_pos_embed,
            past_kv_cache_entry=past_kv_cache_entry,
            attention_mask=attention_mask)
        resid_post = self.step_mlp_resid_post(resid_pre, attn_out)

        return resid_post

    def step_resid_pre(self, resid_pre):
        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]
        return resid_pre
    
    def step_attn_in(self, resid_pre, shortformer_pos_embed):
        if self.cfg.use_attn_in or self.cfg.use_split_qkv_input:
            # We're adding a head dimension
            if shortformer_pos_embed is not None:
                shortformer_pos_embed = utils.repeat_along_head_dimension(
                    shortformer_pos_embed, n_heads=self.cfg.n_heads
                )
        else:
            attn_in = resid_pre

        if self.cfg.use_attn_in:
            attn_in = self.hook_attn_in(
                utils.repeat_along_head_dimension(resid_pre, n_heads=self.cfg.n_heads)
            )
        return attn_in

    def step_qkv_input(self, resid_pre, attn_in):
        if self.cfg.use_split_qkv_input:
            n_kv_heads = (
                self.cfg.n_key_value_heads
                if self.cfg.n_key_value_heads is not None
                and not self.cfg.ungroup_grouped_query_attention
                else self.cfg.n_heads
            )
            query_input = self.hook_q_input(
                utils.repeat_along_head_dimension(resid_pre, n_heads=self.cfg.n_heads)
            )
            key_input = self.hook_k_input(
                utils.repeat_along_head_dimension(resid_pre, n_heads=n_kv_heads)
            )
            value_input = self.hook_v_input(
                utils.repeat_along_head_dimension(resid_pre, n_heads=n_kv_heads)
            )
        else:
            query_input = attn_in
            key_input = attn_in
            value_input = attn_in

        return query_input, key_input, value_input

    def step_attn_out(self, query_input, key_input, value_input,
        shortformer_pos_embed=None,
        past_kv_cache_entry=None,
        attention_mask=None,
    ):
        attn_out = (
            # hook the residual stream states that are used to calculate the
            # queries, keys and values, independently.
            # Then take the layer norm of these inputs, and pass these to the attention module.
            self.attn(
                query_input=self.ln1(query_input)
                + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
                key_input=self.ln1(key_input)
                + (0.0 if shortformer_pos_embed is None else shortformer_pos_embed),
                value_input=self.ln1(value_input),
                past_kv_cache_entry=past_kv_cache_entry,
                attention_mask=attention_mask,
            )
        )  # [batch, pos, d_model]
        if self.cfg.use_normalization_before_and_after:
            # If we use LayerNorm both before and after, then apply the second LN after the layer
            # and before the hook. We do it before the hook so hook_attn_out captures "that which
            # is added to the residual stream"
            attn_out = self.ln1_post(attn_out)
        attn_out = self.hook_attn_out(attn_out)
        return attn_out

    def step_resid_post(self, resid_mid, mlp_out):
        return self.hook_resid_post(resid_mid + mlp_out)
    
    def step_resid_mid(self, resid_pre, attn_out):
        resid_mid = self.hook_resid_mid(resid_pre + attn_out)
        return resid_mid
    
    def step_mlp_in(self, resid_mid):
        mlp_in = (
            resid_mid if not self.cfg.use_hook_mlp_in else self.hook_mlp_in(resid_mid.clone())
        )
        return mlp_in

    def step_mlp_resid_post(self, resid_pre, attn_out):
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            resid_mid = self.step_resid_mid(resid_pre, attn_out)
            mlp_in = self.step_mlp_in(resid_mid)
            normalized_resid_mid = self.ln2(mlp_in)
            mlp_out = self.apply_mlp(normalized_resid_mid)
            resid_post = self.step_resid_post(resid_mid, mlp_out)  # [batch, pos, d_model]
        else:
            raise NotImplementedError('Not yet')

        return resid_post

def apply_mask(x, mask_logits, deterministic=False):
    if deterministic:
        mask = torch.where(mask_logits > 0., 1., 0.)
    else:
        mask = gumbel_sigmoid(mask_logits)
    masked_x = einops.einsum(x, mask,
        "batch query_pos n_heads d_model, n_heads -> batch query_pos n_heads d_model")

    return masked_x

def head_attn_forward(self, query_input, key_input, value_input,
        attention_head_mask=None,
        past_kv_cache_entry = None,
        additive_attention_mask = None,
        attention_mask = None,
        position_bias = None,
    ):
    z = attn_forward(self, query_input, key_input, value_input,
        past_kv_cache_entry,
        additive_attention_mask,
        attention_mask,
        position_bias,
        return_z = True,
    )
    if self.mask_cfg.run_with_mask:
        masked_z = apply_mask(z, self.mask_cfg.mask[self.block_index], True)
    elif self.mask_cfg.use_attention_head_mask:
        masked_z = apply_mask(z, self.attention_head_mask, self.mask_cfg.use_deterministic_mask)
    else:
        masked_z = z

    attn_out = einops.einsum(masked_z, self.W_O,
        "batch query_pos n_heads d_head, n_heads d_head d_model -> batch query_pos d_model") + (self.b_O)
    return attn_out

class HeadTransformerBlock(TransformerBlock):

    def init_masks(self):
        self.attn.attention_head_mask = torch.nn.Parameter(
            torch.nn.init.normal_(torch.ones((self.cfg.n_heads,)).to(self.cfg.device), mean=0.0, std=0.01), 
            requires_grad=True)

    def copy_components(self):
        super().copy_components()
        self.attn.forward = partial(head_attn_forward, self=self.attn)
        self.attn.block_index = self.block_index
        self.attn.mask_cfg = self.mask_cfg

class MaskedTransformerBlock(TransformerBlock):
    def step_resid_mid(self, resid_pre, attn_out):
        resid_mid = self.hook_resid_mid(torch.cat((resid_pre, attn_out), dim=2))  # [batch, pos, d_model]
        return resid_mid

    def step_mlp_in(self, resid_mid):
        sampled_edge_mask_mlp = self.edge_mask_mlp_mask
        mlp_in = einops.einsum(resid_mid, sampled_edge_mask_mlp,
            "batch position prev_head_idx d_model, prev_head_idx -> batch position d_model")
        return mlp_in

    def step_resid_post(self, resid_mid, mlp_out):
        mlp_out = einops.rearrange(mlp_out, "batch position d_model -> batch position 1 d_model")
        resid_post = self.hook_resid_post(torch.cat((resid_mid, mlp_out), dim=2))  # [batch, pos, d_model]
        return resid_post

    def copy_components(self):
        super().copy_components()
    
        self.attn.calculate_qkv_matrices = lambda q,k,v: calculate_qkv_matrices(self.attn, q, k, v)
        self.attn.forward = partial(attn_forward, self=self.attn)

class IntMaskTransformerBlock(MaskedTransformerBlock):

    def init_masks(self):

        prev_nodes = (self.cfg.n_heads + 1) * self.block_index + 1

        self.edge_mask_attention_q_mask = torch.ones(
            (prev_nodes, self.cfg.n_heads), device=self.device)
        self.edge_mask_attention_k_mask = torch.ones(
            (prev_nodes, self.cfg.n_heads), device=self.device)
        self.edge_mask_attention_v_mask = torch.ones(
            (prev_nodes, self.cfg.n_heads), device=self.device)
        self.edge_mask_mlp_mask = torch.ones(
            (prev_nodes + self.cfg.n_heads,), device=self.device)

        self.edge_mask_attention_q_index = {'input': 0}
        self.edge_mask_attention_k_index = {'input': 0}
        self.edge_mask_attention_v_index = {'input': 0}
        self.edge_mask_mlp_index = {'input': 0}

        # For attention masks (qkv), the second dim is always going to be attn head
        for i in range(self.block_index):
            for j in range(self.cfg.n_heads):
                self.edge_mask_attention_q_index[f'{i}.attn_{j}'] = i * (self.cfg.n_heads + 1) + j + 1
                self.edge_mask_attention_k_index[f'{i}.attn_{j}'] = i * (self.cfg.n_heads + 1) + j + 1
                self.edge_mask_attention_v_index[f'{i}.attn_{j}'] = i * (self.cfg.n_heads + 1) + j + 1
                self.edge_mask_mlp_index[f'{i}.attn_{j}'] = i * (self.cfg.n_heads + 1) + j + 1
            self.edge_mask_attention_q_index[f'{i}.mlp'] = i * (self.cfg.n_heads + 1) + j + 2
            self.edge_mask_attention_k_index[f'{i}.mlp'] = i * (self.cfg.n_heads + 1) + j + 2
            self.edge_mask_attention_v_index[f'{i}.mlp'] = i * (self.cfg.n_heads + 1) + j + 2
            self.edge_mask_mlp_index[f'{i}.mlp'] = i * (self.cfg.n_heads + 1) + j + 2

        for j in range(self.cfg.n_heads):
            self.edge_mask_mlp_index[f'{self.block_index}.attn_{j}'] = prev_nodes + j

    def get_qkv_masks(self):
        
        edge_mask_q = self.edge_mask_attention_q_mask
        edge_mask_k = self.edge_mask_attention_k_mask
        edge_mask_v = self.edge_mask_attention_v_mask

        return edge_mask_q, edge_mask_k, edge_mask_v

    def step_attn_out(self, query_input, key_input, value_input,
        shortformer_pos_embed=None,
        past_kv_cache_entry=None,
        attention_mask=None,
    ):
        edge_mask_q, edge_mask_k, edge_mask_v = self.get_qkv_masks()

        masked_query_input = einops.einsum(
            query_input, edge_mask_q,
            "batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model"
        )
        masked_key_input   = einops.einsum(
            key_input, edge_mask_k,
            "batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model"
        )
        masked_value_input = einops.einsum(
            value_input, edge_mask_v,
            "batch position prev_head_idx d_model, prev_head_idx n_heads -> batch position n_heads d_model"
        )

        return super().step_attn_out(masked_query_input, masked_key_input, masked_value_input, 
            shortformer_pos_embed=shortformer_pos_embed,
            past_kv_cache_entry=past_kv_cache_entry,
            attention_mask=attention_mask
        )

    def get_weights(self):
        masks = {
            'q': self.edge_mask_attention_q_mask,
            'k': self.edge_mask_attention_k_mask,
            'v': self.edge_mask_attention_v_mask,
            'mlp': self.edge_mask_mlp_mask,
        }

        return masks

    def _set_edge(self, from_node, to_node, value):
        layer, node = from_node.split('.')
        if node == 'mlp':
            index = self.edge_mask_mlp_index[to_node]
            self.edge_mask_mlp_mask[index] = value
        else:
            node, head = node.split('_')
            head = int(head)
            if node == 'q':
                mask = self.edge_mask_attention_q_mask
                index = self.edge_mask_attention_q_index[to_node]
            elif node == 'k':
                mask = self.edge_mask_attention_k_mask
                index = self.edge_mask_attention_k_index[to_node]
            elif node == 'v':
                mask = self.edge_mask_attention_v_mask
                index = self.edge_mask_attention_v_index[to_node]
            else:
                raise ValueError
            mask[index][head] = value

    def get_edge(self, from_node, to_node):
        layer, node = from_node.split('.')
        if node == 'mlp':
            index = self.edge_mask_mlp_index[to_node]
            return self.edge_mask_mlp_mask[index]
        else:
            node, head = node.split('_')
            head = int(head)
            if node == 'q':
                mask = self.edge_mask_attention_q_mask
                index = self.edge_mask_attention_q_index[to_node]
            elif node == 'k':
                mask = self.edge_mask_attention_k_mask
                index = self.edge_mask_attention_k_index[to_node]
            elif node == 'v':
                mask = self.edge_mask_attention_v_mask
                index = self.edge_mask_attention_v_index[to_node]
            else:
                raise ValueError
            return mask[index][head]

    def get_connections(self, node):
        _, _node = node.split('.')
        if _node == 'mlp':
            return list(self.edge_mask_mlp_index.keys())

        _node, head = _node.split('_')
        if _node == 'q':
            return list(self.edge_mask_attention_q_index.keys())
        elif _node == 'k':
            return list(self.edge_mask_attention_k_index.keys())
        elif _node == 'v':
            return list(self.edge_mask_attention_v_index.keys())
        else:
            raise ValueError

class CircuitTransformerBlock(MaskedTransformerBlock):

    def init_masks(self):

        prev_nodes = (self.cfg.n_heads + 1) * self.block_index + 1

        self.edge_mask_attention_q_logits = torch.nn.Parameter(
            torch.nn.init.normal_(
                torch.ones((prev_nodes, self.cfg.n_heads)),
                mean=self.mask_cfg.edge_hparams.logits_init, std=0.01), 
            requires_grad=True)

        self.edge_mask_attention_k_logits = torch.nn.Parameter(
            torch.nn.init.normal_(
                torch.ones((prev_nodes, self.cfg.n_heads)),
                mean=self.mask_cfg.edge_hparams.logits_init, std=0.01), 
            requires_grad=True)

        self.edge_mask_attention_v_logits = torch.nn.Parameter(
            torch.nn.init.normal_(
                torch.ones((prev_nodes, self.cfg.n_heads)),
                mean=self.mask_cfg.edge_hparams.logits_init, std=0.01), 
            requires_grad=True)

        self.edge_mask_mlp_logits = torch.nn.Parameter(
            torch.nn.init.normal_(
                torch.ones((prev_nodes + self.cfg.n_heads,)),
                mean=self.mask_cfg.edge_hparams.logits_init, std=0.01), 
            requires_grad=True)

    def get_qkv_masks(self):
        if self.mask_control.use_edge_masks:
            if self.mask_control.deterministic:
                edge_mask_q = torch.where(self.edge_mask_attention_q_logits > 0., 1., 0.)
                edge_mask_k = torch.where(self.edge_mask_attention_k_logits > 0., 1., 0.)
                edge_mask_v = torch.where(self.edge_mask_attention_v_logits > 0., 1., 0.)
            else:
                edge_mask_q = gumbel_sigmoid(self.edge_mask_attention_q_logits, self.mask_cfg.edge_hparams.gs_temp)
                edge_mask_k = gumbel_sigmoid(self.edge_mask_attention_k_logits, self.mask_cfg.edge_hparams.gs_temp)
                edge_mask_v = gumbel_sigmoid(self.edge_mask_attention_v_logits, self.mask_cfg.edge_hparams.gs_temp)

            if self.mask_control.reverse:
                edge_mask_q = 1. - edge_mask_q
                edge_mask_k = 1. - edge_mask_k
                edge_mask_v = 1. - edge_mask_v
        else:
            edge_mask_q = torch.ones_like(self.edge_mask_attention_q_logits)
            edge_mask_k = torch.ones_like(self.edge_mask_attention_k_logits)
            edge_mask_v = torch.ones_like(self.edge_mask_attention_v_logits)

        return edge_mask_q, edge_mask_k, edge_mask_v
