import json
import itertools
import math
import random
from pathlib import Path
from types import SimpleNamespace as Namespace
from tqdm.auto import tqdm

import nltk.corpus

from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
import transformer_lens.utils as tl_utils
from datasets import Dataset

TEMPLATE_CTX = (
    "{context_str} "
    "{query_str}"
)

TEMPLATE_NCTX = (
    "{query_str}"
)

def unroll_dict(entries):
    dic = {key:[] for key in entries[0].keys()}
    for entry in entries:
        for k,v in entry.items():
            dic[k].append(v)
    return dic

class LREDataset:

    stop_names = ['superclass', 'substance', 'word', 'person', 'band', 'year', 'pokemon', 'superhero']
    stop_relation_types = ['bias', 'linguistic']

    irr_context_count = 1000
    prompts_count = 10000
    random_word_counts = 100

    def __init__(self, path, tokenizer):
        self.path = path
        self.tokenizer = tokenizer
        self.relations = self.read_relations()

        self.random_words = self.get_random_words()

    def get_token_id(self, token):
        if 'llama-3.1' in self.tokenizer.name_or_path.lower():
            return self.tokenizer.encode(' '+token)[1]
        elif 'gpt2' in self.tokenizer.name_or_path.lower():
            return self.tokenizer.encode(' '+token)[0]
        elif 'llama-2' in self.tokenizer.name_or_path.lower():
            return self.tokenizer.encode(token)[1]

    def get_random_words(self):
        random_words = list(set(nltk.corpus.abc.words()))[:self.random_word_counts]
        return random_words

    def domains(self):
        d = []
        for relation in self.relations.values():
            d.append(relation['properties']['domain_name'])
            d.append(relation['properties']['range_name'])
        return sorted(list(set(d)))

    def read_relations(self):
        relations = {}
        for file_path in self.path.glob('**/*.json'):
            with open(file_path) as open_file:
                data = json.load(open_file)

            if data['properties']['relation_type'] in self.stop_relation_types:
                continue
            if data['properties']['domain_name'] in self.stop_names:
                continue
            if data['properties']['range_name'] in self.stop_names:
                continue

            relations[data['name']] = data
        return relations

    def get_data(self, relation, setting):
        entries = getattr(self, 'get_data_' + setting)(relation)
        data_dict = unroll_dict(entries)
        return Dataset.from_dict(data_dict) 

    def get_headsearch_data(self, relation, no_split=False):
        data = self.relations[relation]

        def get_ds(ss):
            combs = []
            prompts = []
            for sample in ss:
                for ctx_sample in data['samples']:
                    subject = sample['subject']
                    object = sample['object']

                    ctx_subject = ctx_sample['subject']
                    ctx_object = ctx_sample['object']

                    if object != ctx_object:
                        combs.append((subject, object, ctx_subject, ctx_object))

            if len(combs) > 100:
                combs = random.sample(combs, 100)

            for prompt_templates_zs in data['prompt_templates_zs']:
                for prompt_templates in data['prompt_templates']:
                    for subject, object, ctx_subj, ctx_object in combs:
                        context = prompt_templates.format(ctx_subj) + ' ' + ctx_object + '.'
                        query = prompt_templates_zs.format(subject)

                        prompts.append(
                            {
                                'wctx_prompt': TEMPLATE_CTX.format(context_str=context, query_str=query),
                                'nctx_prompt': TEMPLATE_NCTX.format(query_str=query),
                                'dstr_token': ctx_object,
                                'dstr_token_idx': self.get_token_id(ctx_object),
                                'gold_token': object,
                                'gold_token_idx': self.get_token_id(object),
                            }
                        )

            data_dict = unroll_dict(prompts)
            return Dataset.from_dict(data_dict).with_format("torch")

        if no_split:
            return get_ds(data['samples'])

        train_samples, test_samples = train_test_split(data['samples'], test_size=0.1)

        td_ds = get_ds(train_samples).train_test_split(test_size=0.1, shuffle=True)
        test_ds = get_ds(test_samples)

        return Namespace(
            train=td_ds['train'],
            dev=td_ds['test'],
            test=test_ds,
        )