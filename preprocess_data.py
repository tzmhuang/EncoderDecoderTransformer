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

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

import sources

sys.path.append('subword-nmt')
from learn_bpe import learn_bpe
from apply_bpe import BPE

TASK = ['de-en', 'fr-en', 'zh-en']
LANG = ['en', 'de', 'fr']
LANG_MODEL = {'en': 'en_core_web_sm',
              'de': 'de_core_news_sm', 'fr': 'fr_core_news_sm'}


special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


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
            yield line.strip().lower()  # strip \n


def download_and_extract(download_dir, url, src_filename, trg_filename):
    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)


    if src_path and trg_path:
        sys.stderr.write(f"Already downloaded and extracted {url}.\n")
        return src_path, trg_path

    compressed_file = _download_file(download_dir, url)

    sys.stderr.write(f"Extracting {compressed_file}.\n")
    with tarfile.open(download_dir+'/'+compressed_file, "r:gz") as corpus_tar:
        corpus_tar.extractall(download_dir)

    src_path = file_exist(download_dir, src_filename)
    trg_path = file_exist(download_dir, trg_filename)

    if src_path and trg_path:
        return src_path, trg_path

    raise OSError(
        f"Download/extraction failed for url {url} to path {download_dir}")


def _download_file(download_dir, url):
    filename = url.split("/")[-1]
    if file_exist(download_dir, filename):
        sys.stderr.write(f"Already downloaded: {url} (at {filename}).\n")
    else:
        sys.stderr.write(f"Downloading from {url} to {filename}.\n")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(
                url, filename=download_dir+'/'+filename, reporthook=t.update_to)
    return filename


def file_exist(dir_name, file_name):
    for sub_dir, _, files in os.walk(dir_name):
        if file_name in files:
            return os.path.join(sub_dir, file_name)
    return None


def get_source_files(source_dir, data_source, task, src, trg):
    raw_files = {"src": [], "trg": [], }
    for d in data_source['data'][task]:
        if data_source['split'] == 'test':
            _src_file = d['file'] + '-' + ''.join(task.split('-')) + '-src.'+src+'.sgm'
            _trg_file = d['file'] + '-' + ''.join(task.split('-')) + '-ref.'+trg+'.sgm'
        else:
            _src_file = d['file'] + '.' + src
            _trg_file = d['file'] + '.' + trg

        src_file, trg_file = download_and_extract(
            source_dir, d['url'], _src_file, _trg_file)
        raw_files["src"].append(src_file)
        raw_files["trg"].append(trg_file)
    return raw_files

def process_sgm_files(raw_dir, fpath):

    pfpath = os.path.join(raw_dir, fpath.split("/")[-1] + '.processed')

    if os.path.isfile(pfpath):
        sys.stderr.write(f"Proccessed file found, skipping.\n")
        return pfpath
    
    sys.stderr.write(f"Processing: {fpath}.\n")

    CMD= ''' grep '<seg id' {} | \\
        sed -e 's/<seg id="[0-9]*">\s*//g' | \\
        sed -e 's/\s*<\/seg>\s*//g' | \\
        sed -e "s/\’/\'/g" > {}
        '''.format(fpath, pfpath)

    subprocess.call(CMD, shell=True)
    return pfpath


def compile_files(raw_dir, raw_files, prefix):
    src_fpath = os.path.join(raw_dir, f"raw-{prefix}.src")
    trg_fpath = os.path.join(raw_dir, f"raw-{prefix}.trg")

    if os.path.isfile(src_fpath) and os.path.isfile(trg_fpath):
        sys.stderr.write(f"Merged files found, skip the merging process.\n")
        return src_fpath, trg_fpath

    sys.stderr.write(f"Merge files into two files: {src_fpath} and {trg_fpath}.\n")

    with open(src_fpath, 'w') as src_outf, open(trg_fpath, 'w') as trg_outf:
        for src_inf, trg_inf in zip(raw_files['src'], raw_files['trg']):
            sys.stderr.write(f'  Input files: \n'\
                    f'    - SRC: {src_inf}, and\n' \
                    f'    - TRG: {trg_inf}.\n')
            with open(src_inf, newline='\n') as src_inf, open(trg_inf, newline='\n') as trg_inf:
                cntr = 0
                for i, line in enumerate(src_inf):
                    cntr += 1
                    src_outf.write(line.replace('\r', ' ').strip() + '\n')
                for j, line in enumerate(trg_inf):
                    cntr -= 1
                    trg_outf.write(line.replace('\r', ' ').strip() + '\n')
                assert cntr == 0, 'Number of lines in two files are inconsistent.'
    return src_fpath, trg_fpath


def create_bpe_codes(train_files, bpe_codes, code_size, min_freq, verbose=True):
    # merge train files
    tmp_file = "./train_src_trg.tmp"

    if os.path.isfile(bpe_codes):
        sys.stderr.write(f"Code file found, skipping BPE code generation.\n")
        return

    with open(tmp_file, 'w') as f:
        for _file in train_files:
            with open(_file, newline='\n') as infile:
                for i, line in enumerate(infile):
                    f.write(line + '\n')
    with codecs.open(tmp_file, encoding='utf-8') as in_f:
        with codecs.open(bpe_codes, 'w',encoding='utf-8') as out_f:
            learn_bpe(in_f, out_f, code_size, min_freq, verbose)
    os.remove(tmp_file)
    return

def bpe_encode(bpe, infile, outfile):
    sys.stderr.write(f"Endcoding {infile} --> {outfile}\n")

    with codecs.open(infile, encoding='utf-8') as in_f:
        with codecs.open(outfile, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                out_f.write(bpe.process_line(line))
    return

def bpe_encode_files(bpe, src_in, trg_in, data_dir, name, src, trg):
    src_out = os.path.join(data_dir, name + '.' + src)
    trg_out = os.path.join(data_dir, name + '.' + trg)
    
    if os.path.isfile(src_out) and os.path.isfile(trg_out):
        sys.stderr.write(f"Encode file found, skipping BPE encoding.\n")
        return src_out, trg_out
    
    bpe_encode(bpe, src_in, src_out)
    bpe_encode(bpe, trg_in, trg_out)
    return src_out, trg_out
    
def copy_files(src_in, trg_in, data_dir, name, src, trg):
    src_out = os.path.join(data_dir, name + '.' + src)
    trg_out = os.path.join(data_dir, name + '.' + trg)
    
    if os.path.isfile(src_out) and os.path.isfile(trg_out):
        sys.stderr.write(f"{src_out} and {trg_out} found, skip copying.\n")
        return src_out, trg_out
    shutil.copyfile(src_in, src_out)
    shutil.copyfile(trg_in, trg_out)
    return src_out, trg_out


def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def main():
    # Run By:
    #
    # python preprocess_data.py --src_lang en --trg_lang de --data_dir ./wmt17_en_de --save_dir ./wmt17_en_de_processed.pkl --share_vocab
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_lang', required=True, choices=LANG)
    parser.add_argument('--trg_lang', required=True, choices=LANG)
    parser.add_argument('--dataset', type=str, default='wmt17')
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--train_prefix', type=str, default='train')
    parser.add_argument('--test_prefix', type=str, default='test')
    parser.add_argument('--valid_prefix', type=str, default='valid')
    parser.add_argument('--min_freq', type=int, default=1)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--share_vocab', action='store_true')
    parser.add_argument('--use_bpe', action='store_true')
    parser.add_argument('--bpe_codes', type=str, default='bpe_codes')
    parser.add_argument('--code_size', type=int, default=40000, help="BPE code size")
    parser.add_argument('--sep', type=str, default="@@", help="BPE code seperator")


    args = parser.parse_args()
    src = args.src_lang
    trg = args.trg_lang
    task = src + "-" + trg
    if task not in TASK:
        task = trg + '-' + src
    assert task in TASK

    print("Task: ", task)
    # Create folder if needed.
    args.source_dir = './'+'_'.join([args.dataset, src, trg])
    mkdir_if_needed(args.source_dir)
    mkdir_if_needed(args.data_dir)

    # Download and extract raw data.
    raw_train = get_source_files(
        args.source_dir, sources.TRAIN_DATA_SOURCE, task, src, trg)
    raw_val = get_source_files(
        args.source_dir, sources.VAL_DATA_SOURCE, task, src, trg)
    raw_test = get_source_files(
        args.source_dir, sources.TEST_DATA_SOURCE, task, src, trg)

    # process data in .sgm format
    for s in ['src', 'trg']:
        for idx, f in enumerate(raw_test[s]):
            if f.split('/')[-1].split('.')[-1] == 'sgm':
                raw_test[s][idx] = process_sgm_files(args.source_dir, raw_test[s][idx])

    # Merge files together
    train_src, train_trg = compile_files(args.source_dir, raw_train, prefix=task+'-train')
    test_src, test_trg = compile_files(args.source_dir, raw_test, prefix=task+'-test')
    val_src, val_trg = compile_files(args.source_dir, raw_val, prefix=task+'-val')

    tokenizers = {}
    if args.use_bpe:
        args.bpe_codes = os.path.join(args.data_dir, args.bpe_codes)
        create_bpe_codes([train_src, train_trg], args.bpe_codes, args.code_size, args.min_freq, True)
        print("BPE code generation finished")
        print("Building BPE tokenizer")
        with codecs.open(args.bpe_codes, encoding='utf-8') as codes:
            bpe = BPE(codes, separator=args.sep)

        bpe_encode_files(bpe, train_src, train_trg, args.data_dir, 'train', src, trg)
        bpe_encode_files(bpe, test_src, test_trg, args.data_dir, 'test', src, trg)
        bpe_encode_files(bpe, val_src, val_trg, args.data_dir, 'valid', src, trg)

        # using simple split function as tokenizer
        tokenizers[src] = str.split
        tokenizers[trg] = str.split
    else:
        # do something: save data to correct format
        copy_files(train_src, train_trg, args.data_dir, 'train', src, trg)
        copy_files(test_src, test_trg, args.data_dir, 'test', src, trg)
        copy_files(val_src, val_trg, args.data_dir, 'valid', src, trg)

        tokenizers[src] = get_tokenizer('spacy', language=LANG_MODEL[src])
        tokenizers[trg] = get_tokenizer('spacy', language=LANG_MODEL[trg])
        pass

    train_file = {}
    test_file = {}
    valid_file = {}
    for ln in [src, trg]:
        # [train/test/valid].[<src>/<trg>]
        train_file[ln] = args.data_dir + '/' + args.train_prefix + '.' + ln
        test_file[ln] = args.data_dir + '/' + args.test_prefix + '.' + ln
        valid_file[ln] = args.data_dir + '/' + args.valid_prefix + '.' + ln

    # print('Done')
    # return
    # # testing

    print("Generating vocab")
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
    print("Source vocab size: ", len(src_vocab),
          " Target vocab size", len(trg_vocab))

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
