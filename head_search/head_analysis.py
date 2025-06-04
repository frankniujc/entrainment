import json
import itertools
import argparse
import random
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset

from .circuit_lms import MaskedHeadTransformer

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def report(result):
    for key in ['mask density', 'avg logit delta', 'avg prob delta', 'kl_div_wctx', 'kl_div_nctx']:
        print(key+':', result[key])

def kl_div(logits_1, logits_2):
    return F.kl_div(
        logits_1.log_softmax(dim=-1),
        logits_2.log_softmax(dim=-1),
        log_target=True, reduction="none"
    ).sum(dim=-1).mean().item()

def parse_data(path, model):
    td_data, td_golds, td_dstrs = [], [], []
    test_data, test_golds, test_dstrs = [], [], []
    template = (
        "CONTEXT: {context_str} "
        "PROMPT: {prompt_str}"
    )
    template_nctx = "{prompt_str}"

    with open(path) as countries_json:
        countries = json.load(countries_json)

    test_comb = ('Asia', 'Europe')

    for dstr, gold in itertools.permutations(countries, 2):

        if (gold, dstr) == test_comb:
            data, golds, dstrs = test_data, test_golds, test_dstrs
        else:
            data, golds, dstrs = td_data, td_golds, td_dstrs

        contexts = [f'{x} is in {dstr}.' for x in countries[dstr]]
        prompts = [f'{x} is located on the continent of' for x in countries[gold]]

        gold_token_id = model.tokenize_to_id(gold, strict=False)
        dstr_token_id = model.tokenize_to_id(dstr, strict=False)

        for prompt in prompts:
            for context in contexts:
                full = template.format(context_str=context, prompt_str=prompt)
                data.append(full)

                golds.append(gold_token_id)
                dstrs.append(dstr_token_id)

    td_tokenized = model.tokenizer(td_data, return_tensors='pt', padding=True)
    test_tokenized = model.tokenizer(test_data, return_tensors='pt', padding=True)

    test_ds = Dataset.from_dict({
        'input_ids': test_tokenized.input_ids,
        'lengths': test_tokenized.attention_mask.sum(dim=1),
        'golds': torch.LongTensor(test_golds),
        'dstrs': torch.LongTensor(test_dstrs),
    }).with_format("torch")

    ds = Dataset.from_dict({
        'input_ids': td_tokenized.input_ids,
        'lengths': td_tokenized.attention_mask.sum(dim=1),
        'golds': torch.LongTensor(td_golds),
        'dstrs': torch.LongTensor(td_dstrs),
    }).with_format("torch")
    ds = ds.train_test_split(test_size=0.1, shuffle=True)

    return argparse.Namespace(
        train=ds['train'],
        dev=ds['test'],
        test=test_ds,
    )


class HeadAnalysis(MaskedHeadTransformer):
    def setup_exp(self, datasets, batch_size=64):
        self.datasets = datasets

        self.train_dl = torch.utils.data.DataLoader(self.datasets.train, batch_size=batch_size)
        self.dev_dl = torch.utils.data.DataLoader(self.datasets.dev, batch_size=batch_size)
        self.test_dl = torch.utils.data.DataLoader(self.datasets.test, batch_size=batch_size)

        self.n_total_heads = self.cfg.n_layers * self.cfg.n_heads

    def set_mask_cfg(self, use_attention_head_mask=None, run_with_mask=None, use_deterministic_mask=None):
        if use_attention_head_mask is not None:
            self.mask_cfg.use_attention_head_mask = use_attention_head_mask
        if run_with_mask is not None:
            self.mask_cfg.run_with_mask = run_with_mask
        if use_deterministic_mask is not None:
            self.mask_cfg.use_deterministic_mask = use_deterministic_mask

    def set_mask(self, mask):
        self.mask_cfg.run_with_mask = True
        self.mask_cfg.use_deterministic_mask = True
        self.mask_cfg.mask = mask

    def mask_matrix(self):
        _mask_matrix = []
        for layer, mask in self.mask_dict.items():
            _mask = torch.where(mask > 0., 1., 0.)
            _mask_matrix.append(_mask)
        return torch.stack(_mask_matrix)

    def mask_density(self):
        mask_matrix = self.mask_matrix()
        return mask_matrix.sum().item(), self.n_total_heads

    def sparsity_loss(self):
        sparse_loss = 0
        for n, mask_logits in self.mask_dict.items():
            # print(n)
            sparse_loss += F.sigmoid(mask_logits).sum()
        return sparse_loss / self.n_total_heads

    def tokenize_to_id(self, input_token, strict=True):

        input_token = ' ' + input_token

        token = self.tokenizer.tokenize(input_token)
        if strict:
            if len(token) != 1:
                raise ValueError(f'The input token `{input_token}` is tokenized into multiple tokens: {token}.')
        token_id = self.tokenizer.convert_tokens_to_ids(token)[0]
        return token_id


    def prepare_evaluation(self, *dataloaders):
        if not dataloaders:
            dataloaders = (self.dev_dl, self.test_dl)
        for ds in dataloaders:
            self._prepare_evaluation(ds)

    @torch.no_grad
    def _prepare_evaluation(self, dl):

        self.set_mask_cfg(use_attention_head_mask=False)

        original_wctx_logits = []
        original_nctx_logits = []

        for batch in dl:
            wctx_prompts = batch['wctx_prompt']

            bs = torch.arange(len(wctx_prompts))

            tokens = self.tokenizer(wctx_prompts, return_tensors='pt', padding=True)
            logits = self(tokens['input_ids'].to(0))
            lengths = tokens['attention_mask'].sum(dim=1)
            last_token_logits = logits[bs, lengths-1, :]
            original_wctx_logits.append(last_token_logits.cpu())

            nctx_prompts = batch['nctx_prompt']
            tokens = self.tokenizer(nctx_prompts, return_tensors='pt', padding=True)
            logits = self(tokens['input_ids'].to(0))
            lengths = tokens['attention_mask'].sum(dim=1)
            last_token_logits = logits[bs, lengths-1, :]
            original_nctx_logits.append(last_token_logits.cpu())

        dl.original_wctx_logits = torch.cat(original_wctx_logits, dim=0)
        dl.original_nctx_logits = torch.cat(original_nctx_logits, dim=0)


    @torch.no_grad
    def evaluate(self, dl, use_attention_head_mask=True, disable_tqdm=False):

        self.eval()
        self.set_mask_cfg(use_attention_head_mask=use_attention_head_mask, use_deterministic_mask=True)

        logit_diffs = []
        prob_diffs = []
        correct = 0

        logit_lst = []

        for batch in tqdm(dl, disable=disable_tqdm):
            sents = batch['wctx_prompt']
            gold_token = batch['gold_token_idx']
            dstr_token = batch['dstr_token_idx']

            bs = torch.arange(len(sents))
            tokens = self.tokenizer(sents, return_tensors='pt', padding=True)

            logits = self(tokens['input_ids'].to(0))
            lengths = tokens['attention_mask'].sum(dim=1)
            last_token_logits = logits[bs, lengths-1, :]
            last_token_probs = last_token_logits.softmax(1)

            gold_logits = last_token_logits[bs, gold_token]
            dstr_logits = last_token_logits[bs, dstr_token]
            logits_diff = gold_logits - dstr_logits

            gold_probs = last_token_probs[bs, gold_token]
            dstr_probs = last_token_probs[bs, dstr_token]
            probs_diff = gold_probs - dstr_probs

            correct += (gold_probs > dstr_probs).sum().item()
            logit_lst.append(last_token_logits.cpu())

        output_logits = torch.cat(logit_lst, dim=0)

        logit_diffs += logits_diff.tolist()
        prob_diffs += probs_diff.tolist()

        result = {
            'mask density': self.mask_density(),
            'mask': self.mask_matrix().tolist(),
            'avg logit delta': np.average(logit_diffs),
            'avg prob delta': np.average(prob_diffs),
            'acc': correct / len(dl.dataset),
            'correct': correct,
            'kl_div_wctx': kl_div(output_logits, dl.original_wctx_logits),
            'kl_div_nctx': kl_div(output_logits, dl.original_nctx_logits),
        }

        return result
    
    def search(self, output_path, sparsity_lambda=1.0, learning_rate=1.0, n_epochs=500, disable_tqdm=False):
        results = {}

        mask_logits = self.mask_dict.values()
        optimizer = torch.optim.AdamW(mask_logits, lr=learning_rate)

        for epoch in range(n_epochs):

            self.train()
            self.set_mask_cfg(use_attention_head_mask=True, use_deterministic_mask=False)

            pbar = tqdm(self.train_dl, disable=disable_tqdm)

            for batch in pbar:
                sents = batch['wctx_prompt']
                gold_token = batch['gold_token_idx']
                dstr_token = batch['dstr_token_idx']

                # sents = sents.to(0)
                bs = torch.arange(len(sents))

                logits = self(sents)
                last_token_logits = logits[bs, -1, :]

                gold_logits = last_token_logits[bs, gold_token]
                dstr_logits = last_token_logits[bs, dstr_token]
                logits_masked = torch.stack([gold_logits, dstr_logits], -1)  # (B, 2)

                effect_loss = F.cross_entropy(
                    logits_masked,
                    torch.zeros(len(sents)).long().to(logits_masked.device))
                sparsity_loss = self.sparsity_loss()
                loss = effect_loss - sparsity_loss * sparsity_lambda

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.set_description(f'loss: {loss.item():.4f} effect loss: {effect_loss.item():.4f} sparsity loss: {sparsity_loss.item():.4f}')

            dev_result = self.evaluate(self.dev_dl)
            test_result = self.evaluate(self.test_dl)
            
            print('dev:')
            report(dev_result)
            print('test:')
            report(test_result)

            results[epoch] = {
                'dev': dev_result,
                'test': test_result,
            }

            with open(output_path, 'w') as open_file:
                json.dump(results, open_file)
