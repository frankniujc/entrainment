import os
import json
import argparse
from argparse import Namespace
from pathlib import Path

import nltk
nltk.data.path.append('/mnt/beegfs/work/niu/codes/heads/data')

import torch
import torch.nn.functional as F
import numpy as np

from tqdm.auto import tqdm

from head_search.head_analysis import HeadAnalysis
from head_search.data.lre_dataset import LREDataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--relation', type=str)
    parser.add_argument('-f', '--output-filename', type=str, default=None)
    parser.add_argument('-o', '--output-dir', type=Path, default=Path('output'))
    parser.add_argument('-m', '--model', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--disable-tqdm', action='store_true')
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    args = parser.parse_args()

    relation = args.relation.replace('_', ' ')

    if args.output_filename is None:
        args.output_filename = f'{(args.model).split('/')[-1]}--{args.relation}.json'

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = HeadAnalysis.from_pretrained(args.model)

    mask_cfg = Namespace(
        use_attention_head_mask=True,
        use_deterministic_mask=False,
        run_with_mask=False,
        mask=None,
    )
    model.setup_mask_param(mask_cfg)
    
    lre_path = Path('data/lre_dataset')
    lre_data = LREDataset(lre_path, model.tokenizer)
    model.setup_exp(lre_data.get_headsearch_data(relation), args.batch_size)
    model.prepare_evaluation()

    model.search(args.output_dir / args.output_filename, disable_tqdm=args.disable_tqdm)