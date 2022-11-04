import sys
import os
import subprocess
import random
import math
import time
import argparse
import pickle
import copy
import codecs
import shutil

import spacy
from tqdm import tqdm
import tarfile
import urllib
import numpy as np
from collections import Counter
import pandas as pd

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

import sources

sys.path.append('subword-nmt')
from learn_bpe import learn_bpe
from apply_bpe import BPE
from helper import *

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _text_iter(dir):
    # input: text file with one sentence per line
    # output: string iterable
    with open(dir, 'r') as f:
        for line in f:
            yield line.strip().lower()  # strip \n


def extract(source_dir, compressed_file, filename):
    _path = file_exist(source_dir, filename)

    if _path:
        sys.stderr.write(f"Already extracted {compressed_file}.\n")
        return _path

    sys.stderr.write(f"Extracting {compressed_file}.\n")
    with tarfile.open(source_dir+'/'+compressed_file, "r:gz") as corpus_tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(corpus_tar, source_dir)

    _path = file_exist(source_dir, filename)
    if _path:
        return _path

    raise OSError(
        f"Extraction failed for url {compressed_file} to path {source_dir}")


def file_exist(dir_name, file_name):
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None


def get_source_files(source_dir, data_source, task):
    raw_files = []
    for d in data_source['data'][task]:
        if data_source['split'] == 'test':
            filename = task + '_test.tar.gz'
        else:
            filename = task+'.tar.gz'
        _file = extract(
            source_dir, filename, d['file'])
        raw_files.append(_file)
    return raw_files


def compile_files(raw_dir, raw_files, prefix):
    src_fpath = os.path.join(raw_dir, f"raw-{prefix}.src")
    trg_fpath = os.path.join(raw_dir, f"raw-{prefix}.trg")
    scr_fpath = os.path.join(raw_dir, f"raw-{prefix}.score")

    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath) and os.path.isfile(scr_fpath):
        sys.stderr.write(f"Merged files found, skip the merging process.\n")
        return src_fpath, trg_fpath, scr_fpath

    sys.stderr.write(
        f"Merge files into two files: {src_fpath} and {trg_fpath}.\n")

    with open(src_fpath, 'w') as src_outf, open(trg_fpath, 'w') as trg_outf,  open(scr_fpath, 'w') as scr_outf:
        for raw_file in raw_files:
            sys.stderr.write(f'  Input files: SRC: {raw_file}\n')
        tsv_df = pd.read_csv(raw_file, sep='\t')
        for idx in range(tsv_df.shape[0]):
            src = tsv_df.iloc[idx]['original']
            trg = tsv_df.iloc[idx]['translation']
            score = tsv_df.iloc[idx]['z_mean']
            src_outf.write(src.replace('\r', ' ').strip() + '\n')
            trg_outf.write(trg.replace('\r', ' ').strip() + '\n')
            scr_outf.write(str(score) + '\n')
    return src_fpath, trg_fpath, scr_fpath


def bpe_encode(bpe, infile, outfile):
    sys.stderr.write(f"Endcoding {infile} --> {outfile}\n")

    with codecs.open(infile, encoding='utf-8') as in_f:
        with codecs.open(outfile, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                out_f.write(bpe.process_line(line))
    return


def bpe_encode_files(bpe, src_in, trg_in, score_in, data_dir, name, src, trg):
    src_out = os.path.join(data_dir, name + '.' + src)
    trg_out = os.path.join(data_dir, name + '.' + trg)
    score_out = os.path.join(data_dir, name + '.score')

    if os.path.isfile(src_out) and os.path.isfile(trg_out) and os.path.isfile(score_out):
        sys.stderr.write(f"Encode file found, skipping BPE encoding.\n")
        return src_out, trg_out, score_out

    bpe_encode(bpe, src_in, src_out)
    bpe_encode(bpe, trg_in, trg_out)
    shutil.copyfile(score_in, score_out)
    return src_out, trg_out, score_out


def copy_files(src_in, trg_in, score_in, data_dir, name, src, trg):
    src_out = os.path.join(data_dir, name + '.' + src)
    trg_out = os.path.join(data_dir, name + '.' + trg)
    score_out = os.path.join(data_dir, name + '.score')

    if os.path.isfile(src_out) and os.path.isfile(trg_out):
        sys.stderr.write(f"{src_out} and {trg_out} found, skip copying.\n")
        return src_out, trg_out
    shutil.copyfile(src_in, src_out)
    shutil.copyfile(trg_in, trg_out)
    shutil.copyfile(score_in, score_out)
    return src_out, trg_out, score_out


def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', required=True, choices=LANG)
    parser.add_argument('--trg_lang', required=True, choices=LANG)
    parser.add_argument('--source_dir', required=True,
                        type=str, help="Directory for source data")
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--vocab_file', type=str, required=True,
                        help="file generated from preproecss_data.py")
    parser.add_argument('--data_dir', default='mlqe_data', type=str,
                        help="store location for processed data")
    parser.add_argument('--train_prefix', type=str, default='train')
    parser.add_argument('--test_prefix', type=str, default='test')
    parser.add_argument('--valid_prefix', type=str, default='valid')
    parser.add_argument('--max_length', type=int,
                        default=150, help='max length of sentence')
    parser.add_argument('--use_bpe', action='store_true')
    parser.add_argument('--bpe_codes', type=str,
                        default='bpe_codes', help="Path to BPE code file")
    parser.add_argument('--sep', type=str, default="@@",
                        help="BPE code seperator")

    args = parser.parse_args()
    src = args.src_lang
    trg = args.trg_lang
    task = src + "-" + trg
    assert task in TASK

    print("Task: ", task)
    mkdir_if_needed(args.data_dir)
    args.source_dir = args.source_dir + "/data"

    # Extract raw data from source_dir:
    #   - git clone https://github.com/facebookresearch/mlqe.git
    #
    raw_train = get_source_files(
        args.source_dir, sources.QE_TRAIN_DATA_SOURCE, task)
    raw_val = get_source_files(
        args.source_dir, sources.QE_VAL_DATA_SOURCE, task)
    raw_test = get_source_files(
        args.source_dir, sources.QE_TEST_DATA_SOURCE, task)

    # Create raw sentence and score file
    train_src, train_trg, train_score = compile_files(
        args.source_dir, raw_train, prefix=task+'-train')
    test_src, test_trg, test_score = compile_files(
        args.source_dir, raw_test, prefix=task+'-test')
    val_src, val_trg, val_score = compile_files(
        args.source_dir, raw_val, prefix=task+'-val')

    tokenizers = {}
    if args.use_bpe:
        print(f"Looking for BPE code in {args.bpe_codes}")
        with codecs.open(args.bpe_codes, encoding='utf-8') as codes:
            bpe = BPE(codes, separator=args.sep)

        bpe_encode_files(bpe, train_src, train_trg, train_score,
                         args.data_dir, 'train', src, trg)
        bpe_encode_files(bpe, test_src, test_trg, test_score,
                         args.data_dir, 'test', src, trg)
        bpe_encode_files(bpe, val_src, val_trg, val_score,
                         args.data_dir, 'valid', src, trg)

        # using simple split function as tokenizer
        tokenizers[src] = str.split
        tokenizers[trg] = str.split
    else:
        copy_files(train_src, train_trg, train_score,
                   args.data_dir, 'train', src, trg)
        copy_files(test_src, test_trg, test_score,
                   args.data_dir, 'test', src, trg)
        copy_files(val_src, val_trg, val_score,
                   args.data_dir, 'valid', src, trg)

        tokenizers[src] = get_tokenizer('spacy', language=LANG_MODEL[src])
        tokenizers[trg] = get_tokenizer('spacy', language=LANG_MODEL[trg])
        pass

    train_file = {}
    test_file = {}
    valid_file = {}
    for ln in [src, trg, 'score']:
        # [train/test/valid].[<src>/<trg>]
        train_file[ln] = args.data_dir + '/' + args.train_prefix + '.' + ln
        test_file[ln] = args.data_dir + '/' + args.test_prefix + '.' + ln
        valid_file[ln] = args.data_dir + '/' + args.valid_prefix + '.' + ln

    print("Loading vocab")  # load vocab instead of reading
    if not os.path.exists(args.vocab_file):
        print(f"Vocabulary file does not exist: Build Vocabulary by running preprocess_data.py")
        return

    with open(args.vocab_file, 'rb') as f:
        corpus_data = pickle.load(f)

    src_vocab = corpus_data['vocab']['src']
    trg_vocab = corpus_data['vocab']['trg']

    src_vocab.set_default_index(UNK_IDX)
    trg_vocab.set_default_index(UNK_IDX)

    print(f"Finished vocab loading from {args.vocab_file}")
    print("Source vocab size: ", len(src_vocab),
          "Target vocab size: ", len(trg_vocab))

    train = [t for t in zip(_text_iter(train_file[src]), _text_iter(train_file[trg]), _text_iter(train_file['score'])) if (
        len(tokenizers[src](t[0])) <= args.max_length and len(tokenizers[trg](t[1])) <= args.max_length)]
    test = [t for t in zip(_text_iter(test_file[src]), _text_iter(test_file[trg]), _text_iter(test_file['score'])) if (
        len(tokenizers[src](t[0])) <= args.max_length and len(tokenizers[trg](t[1])) <= args.max_length)]
    valid = [t for t in zip(_text_iter(valid_file[src]), _text_iter(valid_file[trg]), _text_iter(valid_file['score'])) if (
        len(tokenizers[src](t[0])) <= args.max_length and len(tokenizers[trg](t[1])) <= args.max_length)]

    data = {'vocab_setting': corpus_data['setting'],
            'setting': args,
            'vocab': {'src': src_vocab, 'trg': trg_vocab},
            'train': train,
            'valid': valid,
            'test': test}

    print("Dumping data to: ", args.save_dir)
    with open(args.save_dir, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
