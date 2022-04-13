import sys
import os
import random
import math
import time
import argparse
import pickle
import copy

import spacy
import numpy as np
from collections import Counter

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

LANG = ['en', 'de', 'fr']
LANG_MODEL = {'en': 'en_core_web_sm',
              'de': 'de_core_news_sm', 'fr': 'fr_core_news_sm'}
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


def _build_vocab(text_iter, tokenizer, min_freq, specials):
    counter = Counter()
    for line in text_iter:
        counter.update(tokenizer(line))
    return vocab(counter, min_freq=min_freq, specials=specials)


def _build_shared_vocab(src_iter, trg_iter, src_tokenizer, trg_tokenizer, min_freq, specials, lower=True):
    counter = Counter()
    for line in src_iter:
        if lower:
            line = line.lower()
        counter.update(src_tokenizer(line))

    for line in trg_iter:
        if lower:
            line = line.lower()
        counter.update(trg_tokenizer(line))
    return vocab(counter, min_freq=min_freq, specials=specials)


def _text_iter(dir):
    # input: text file with one sentence per line
    # output: string iterable
    with open(dir, 'r') as f:
        for line in f:
            yield line.strip()  # strip \n

def main():
    # Run By: 
    # 
    # python preprocess_data.py --src_lang en --trg_lang de --data_dir ./wmt17_en_de --save_dir ./wmt17_en_de_processed.pkl --share_vocab
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', required=True, choices=LANG)
    parser.add_argument('--trg_lang', required=True, choices=LANG)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--train_prefix', type=str, default='train')
    parser.add_argument('--test_prefix', type=str, default='test')
    parser.add_argument('--valid_prefix', type=str, default='valid')
    parser.add_argument('--min_freq', type=int, default=1)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--share_vocab', action='store_true')

    args = parser.parse_args()
    src = args.src_lang
    trg = args.trg_lang

    train_file = {}
    test_file = {}
    valid_file = {}
    for ln in [src, trg]:
        # [train/test/valid].[<src>/<trg>]
        train_file[ln] = args.data_dir + '/' + args.train_prefix + '.' + ln
        test_file[ln] = args.data_dir + '/' + args.test_prefix + '.' + ln
        valid_file[ln] = args.data_dir + '/' + args.valid_prefix + '.' + ln

    tokenizers = {}
    tokenizers[src] = get_tokenizer('spacy', language=LANG_MODEL[src])
    tokenizers[trg] = get_tokenizer('spacy', language=LANG_MODEL[trg])

    MIN_FREQ = args.min_freq
    if args.share_vocab:
        # merge vocab of src and trg
        src_vocab = _build_shared_vocab(_text_iter(train_file[src]), _text_iter(train_file[trg]), tokenizers[src],
                                        tokenizers[trg], min_freq=MIN_FREQ, specials=special_symbols)
        trg_vocab = src_vocab
    else:
        src_vocab = _build_vocab(_text_iter(
            train_file[src]), tokenizers[src], min_freq=MIN_FREQ, specials=special_symbols)
        trg_vocab = _build_vocab(_text_iter(
            train_file[trg]), tokenizers[trg], min_freq=MIN_FREQ, specials=special_symbols)

    print("Finished vocab generation")
    print("Source vocab size: ", len(src_vocab), " Target vocab size", len(trg_vocab))

    train = list(zip(_text_iter(train_file[src]), _text_iter(train_file[trg])))
    test = list(zip(_text_iter(test_file[src]), _text_iter(test_file[trg])))
    valid = list(zip(_text_iter(valid_file[src]), _text_iter(valid_file[trg])))


    data = {'setting': args,
            'vocab': {'src': src_vocab, 'trg': trg_vocab},
            'train': train,
            'valid': valid,
            'test': test}

    print("Dumping data to: ", args.save_dir)
    with open(args.save_dir, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
