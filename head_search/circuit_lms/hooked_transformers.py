import os
import logging
from argparse import Namespace
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import transformer_lens.utils as utils
from transformer_lens.utils import USE_DEFAULT_VALUE
from transformer_lens.utilities import devices
from transformer_lens.HookedTransformer import Output

from transformer_lens import HookedTransformer, HookedEncoder
from .transformer_blocks import TransformerBlock, IntMaskTransformerBlock, CircuitTransformerBlock, HeadTransformerBlock, gumbel_sigmoid

class BaseTransformer:

    @property
    def device(self):
        return next(self.parameters()).device

    def setup_mask_param(self, mask_cfg):
        self.mask_cfg = mask_cfg
        self.mask_control = Namespace(
            use_edge_masks = True,
            use_weight_masks = True,
            deterministic = False,
            reverse = False,
        )
        self.blocks = nn.ModuleList(
            [self.transformer_block_cls(block, block_index, mask_cfg, self.mask_control) for block_index, block in enumerate(self.blocks)]
        )

    def forward(
        self,
        input,
        return_type = "logits",
        loss_per_token: bool = False,
        prepend_bos = USE_DEFAULT_VALUE,
        padding_side = USE_DEFAULT_VALUE,
        start_at_layer = None,
        tokens = None,
        shortformer_pos_embed = None,
        attention_mask = None,  # [batch pos]
        stop_at_layer = None,
        past_kv_cache = None,
    ):
        
        with utils.LocallyOverridenDefaults(
            self, prepend_bos=prepend_bos, padding_side=padding_side
        ):
            if start_at_layer is None:
                (
                    residual,
                    tokens,
                    shortformer_pos_embed,
                    attention_mask,
                ) = self.input_to_embed(
                    input,
                    prepend_bos=prepend_bos,
                    padding_side=padding_side,
                    attention_mask=attention_mask,
                    past_kv_cache=past_kv_cache,
                )
            else:
                assert type(input) == torch.Tensor
                residual = input

            if start_at_layer is None:
                start_at_layer = 0

            blocks_and_idxs = list(zip(range(self.cfg.n_layers), self.blocks))

            residual = self.step_residual_input(residual)

            for i, block in blocks_and_idxs[start_at_layer:stop_at_layer]:
                residual = residual.to(devices.get_device_for_block_index(i, self.cfg))
                if shortformer_pos_embed is not None:
                    shortformer_pos_embed = shortformer_pos_embed.to(
                        devices.get_device_for_block_index(i, self.cfg)
                    )

                residual = block(
                    residual,
                    past_kv_cache_entry=past_kv_cache[i] if past_kv_cache is not None else None,
                    shortformer_pos_embed=shortformer_pos_embed,
                    attention_mask=attention_mask,
                )  # [batch, pos, d_model]

            if stop_at_layer is not None:
                # When we stop at an early layer, we end here rather than doing further computation
                return residual

            residual = self.step_residual_output(residual)

            if self.cfg.normalization_type is not None:
                residual = self.ln_final(residual)  # [batch, pos, d_model]
            if return_type is None:
                return None
            else:
                logits = self.unembed(residual)  # [batch, pos, d_vocab]
                if self.cfg.output_logits_soft_cap > 0.0:
                    logits = self.cfg.output_logits_soft_cap * F.tanh(
                        logits / self.cfg.output_logits_soft_cap
                    )
                if return_type == "logits":
                    return logits
                else:
                    assert (
                        tokens is not None
                    ), "tokens must be passed in if return_type is 'loss' or 'both'"
                    loss = self.loss_fn(logits, tokens, attention_mask, per_token=loss_per_token)
                    if return_type == "loss":
                        return loss
                    elif return_type == "both":
                        return Output(logits, loss)
                    else:
                        logging.warning(f"Invalid return_type passed in: {return_type}")
                        return None

    def step_residual_input(self, residual):
        return residual

    def step_residual_output(self, residual):
        return residual

class TLTransformer(BaseTransformer, HookedTransformer):
    '''TransformerLens HookedTransformer'''
    transformer_block_cls = TransformerBlock

class MaskedHeadTransformer(BaseTransformer, HookedTransformer):
    transformer_block_cls = HeadTransformerBlock

    def setup_mask_param(self, mask_cfg):
        super().setup_mask_param(mask_cfg)

        self.mask_dict = {}
        for block in self.blocks:
            self.mask_dict[block.block_index] = block.attn.attention_head_mask

        for name, p in self.named_parameters():
            if 'attention_head_mask' in name:
                pass
            else:
                p.grad = None
                p.requires_grad = False

class MaskedTransformer(BaseTransformer, HookedTransformer):

    def step_residual_input(self, residual):
        residual = einops.rearrange(residual, "batch position d_model -> batch position 1 d_model")
        return residual

    def step_residual_output(self, residual):
        sampled_output_mask = self.edge_mask_output_mask
        residual = einops.einsum(
            residual, sampled_output_mask,
            "batch position prev_head_idx d_model, prev_head_idx -> batch position d_model"
        )
        return residual

class IntMaskTransformer(MaskedTransformer, HookedTransformer):
    transformer_block_cls = IntMaskTransformerBlock

    def setup_mask_param(self, mask_cfg=None):
        super().setup_mask_param(mask_cfg)

        total_nodes = (self.cfg.n_heads + 1) * self.cfg.n_layers + 1
        self.edge_mask_output_mask = torch.ones((total_nodes,), device=self.device)
        self.edge_mask_output_index = {'input': 0}
        for i in range(self.cfg.n_layers):
            for j in range(self.cfg.n_heads):
                self.edge_mask_output_index[f'{i}.attn_{j}'] = i*(self.cfg.n_heads + 1) + j + 1
            self.edge_mask_output_index[f'{i}.mlp'] = i*(self.cfg.n_heads + 1) + j + 2


    def get_masks(self, flat=True):
        masks = {
            'output': self.edge_mask_output_mask
        }
        for i, block in enumerate(self.blocks):
            if flat:
                for k, v in block.get_weights().items():
                    masks[f'block_{i}_{k}'] = v
            else:
                masks[f'block_{i}'] = block.get_weights()
        return masks
    
    def density(self):
        masks = self.get_masks()
        total = int(sum(x.nelement() for x in masks.values()))
        on = int(sum(x.sum() for x in masks.values()))
        return on/total, on, total
    
    def _set_edge(self, to_node, value):
        index = self.edge_mask_output_index[to_node]
        self.edge_mask_output_mask[index] = value

    def set_edge(self, from_node, to_node, value):
        masks = self.get_masks()

        if from_node == 'output':
            self._set_edge(to_node, value)
        else:
            layer, node = from_node.split('.')
            layer = int(layer)
            self.blocks[layer]._set_edge(from_node, to_node, value)

    def get_edge(self, from_node, to_node):
        masks = self.get_masks()

        if from_node == 'output':
            index = self.edge_mask_output_index[to_node]
            return self.edge_mask_output_mask[index]
        else:
            layer, node = from_node.split('.')
            layer = int(layer)
            return self.blocks[layer].get_edge(from_node, to_node)

    def yield_from_nodes(self):
        '''Topo sorted'''

        yield 'output'
        for i in range(self.cfg.n_layers-1, -1, -1):
            yield f'{i}.mlp'
            for attn_type in 'qkv':
                for j in range(self.cfg.n_heads):
                    yield f'{i}.{attn_type}_{j}'

    def get_incoming_edges(self, to_node):
        results = {}
        for from_node in self.yield_from_nodes():
            try:
                edge = self.get_edge(from_node, to_node)
                results[(from_node, to_node)] = edge.item()
            except IndexError:
                pass
            except KeyError:
                pass
        return results

    def save_masks(self, dir_path):
        masks = self.get_masks()
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(masks, dir_path / 'masks.pt')

    def get_connections(self, node):
        if node == 'output':
            return list(self.edge_mask_output_index.keys())
        layer, _node = node.split('.')
        layer = int(layer)
        return self.blocks[layer].get_connections(node)

class CircuitTransformer(MaskedTransformer, HookedTransformer):
    transformer_block_cls = CircuitTransformerBlock

    def setup_mask_param(self, mask_cfg=None):
        super().setup_mask_param(mask_cfg)

        total_nodes = (self.cfg.n_heads + 1) * self.cfg.n_layers + 1
        self.edge_mask_output_logits = torch.nn.Parameter(
            torch.nn.init.normal_(torch.ones((total_nodes,)), mean=self.mask_cfg.edge_hparams.logits_init, std=0.01), 
            requires_grad=True)

        # initialize mask logits
        # self.mask_logits_device = cfg.mask_logits_device
        self.unmasked_params = {}
        self.mask_logits_dict_weight = {}
        self.mask_logits_dict_edge = {}

        # weight mask logits initialization
        # only register weight mask logits if cfg.use_weight_masks == True to save memory
        # load pretrained mask logits if necessary
        self.n_weight = 0
        for name, p in self.named_parameters():
            # do not learn masks for: 
            # 1) embedding and unembedding layers
            # 2) layernorms
            if 'emb' not in name and 'edge' not in name and 'ln' not in name:  
                p.grad = None
                p.requires_grad = False
                self.unmasked_params[name] = p.clone()

                masks_logits = nn.Parameter(
                    torch.nn.init.normal_(
                        torch.ones_like(p).to('cuda'),
                        mean=self.mask_cfg.weight_hparams.logits_init, std=0.01),  
                    requires_grad=True)
                # we manually put mask_logits onto cuda here, since using nn.ParameterDict will incur an annoying re-naming issue             
                self.mask_logits_dict_weight[name] = masks_logits
                with torch.no_grad():
                    self.n_weight += torch.ones_like(p.view(-1)).sum().cpu()

        # edge mask logits initialization
        self.n_edge = 0
        for name, p in self.named_parameters():
            if 'edge' in name:
                self.mask_logits_dict_edge[name] = p
                with torch.no_grad():
                    self.n_edge += torch.ones_like(p.view(-1)).sum().cpu()

    @property
    def weight_density(self):
        n_weight_preserved = 0
        with torch.no_grad():
            for _, mask in self.mask_logits_dict_weight.items():
                n_weight_preserved += torch.where(mask >= 0., 1, 0).sum()

        weight_den = int(n_weight_preserved.item()) / int(self.n_weight.item())
        return int(self.n_weight.item()), int(n_weight_preserved.item()), weight_den

    @property
    def edge_density(self):
        n_edge_preserved = 0
        with torch.no_grad():
            for _, mask in self.mask_logits_dict_edge.items():
                n_edge_preserved += torch.where(mask >= 0., 1, 0).sum()

        edge_den = int(n_edge_preserved.item()) / int(self.n_edge.item())
        return int(self.n_edge.item()), int(n_edge_preserved.item()), edge_den

    def turn_on_weight_masks(self, deterministic=False, reverse=False):
        for name, param in self.named_parameters():
            if name in self.unmasked_params:
                unmasked_m = self.unmasked_params[name].to(param.device)
                mask_logits = self.mask_logits_dict_weight[name]
                if not deterministic:
                    sampled_masks = gumbel_sigmoid(mask_logits, gs_temp=self.mask_cfg.weight_hparams.gs_temp)
                else:
                    with torch.no_grad():
                        sampled_masks = torch.where(mask_logits > 0., 1., 0.)
                if reverse:
                    sampled_masks = 1. - sampled_masks
                param.copy_(sampled_masks * unmasked_m)
        self.mask_control.use_weight_masks = True

    def turn_off_weight_masks(self):
        for name, param in self.named_parameters():
            if name in self.unmasked_params:            
                unmasked_m = self.unmasked_params[name]
                param.copy_(unmasked_m)
                param.detach_()
        self.mask_control.use_weight_masks = False

    def turn_on_edge_masks(self, deterministic=False, reverse=False):
        self.mask_control.use_edge_masks = True
        self.mask_control.deterministic = deterministic
        self.mask_control.reverse = reverse

    def turn_off_edge_masks(self):
        self.mask_control.use_edge_masks = False