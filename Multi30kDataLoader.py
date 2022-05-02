import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import vocab


import pickle
from helper import *
from collections import Counter


def _build_vocab(data_iter, tokenizer, min_freq, specials):
    counter = Counter()
    for line in data_iter:
        counter.update(tokenizer(line))
    return vocab(counter, min_freq=min_freq, specials=specials)

def _tokenizer(src, trg, use_bpe=False):
    if use_bpe:
        src_tok = str.split
        trg_tok = str.split
    else:
        src_tok = get_tokenizer('spacy', language=LANG_MODEL[src])
        trg_tok = get_tokenizer('spacy', language=LANG_MODEL[trg])
    return src_tok, trg_tok

def build_multi30k_vocab(src, trg, min_freq, specials):
    src_data = []
    trg_data = []
    train_dataset, valid_dataset, test_dataset = Multi30k(split=('train', 'valid', 'test'),language_pair=(src, trg))
    src_data = src_data + [d[0].lower() for d in train_dataset]
    trg_data = trg_data + [d[1].lower() for d in train_dataset]

    src_tokenizer, trg_tokenizer = _tokenizer(src, trg)
    src_vocab = _build_vocab(src_data, src_tokenizer, min_freq, specials)
    trg_vocab = _build_vocab(trg_data, trg_tokenizer, min_freq, specials)
    return src_vocab, trg_vocab
    

class Multi30kDataLoader(DataLoader):
    def __init__(self, src, trg, split, batch_size, num_workers, use_bpe=False, shuffle=True):
        dataset = list(Multi30k(split=(split),language_pair=(src, trg)))
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=self.collate_fn)
        self.src = src
        self.trg = trg
        self.vocab = {}
        self.use_bpe = use_bpe
        self.src_tokenizer, self.trg_tokenizer = self.get_tokenizer()
        self.vocab['src'], self.vocab['trg'] = self.get_vocab(2)
        self.data_size = len(dataset)
        return

    def get_vocab(self, min_freq):
        src_vocab, trg_vocab = build_multi30k_vocab(self.src, self.trg, min_freq, special_symbols)
        src_vocab.set_default_index(UNK_IDX)
        trg_vocab.set_default_index(UNK_IDX)
        return src_vocab, trg_vocab


    def get_tokenizer(self):
        if self.use_bpe:
            src_tok = str.split
            trg_tok = str.split
        else:
            src_tok = get_tokenizer('spacy', language=LANG_MODEL[self.src])
            trg_tok = get_tokenizer('spacy', language=LANG_MODEL[self.trg])
        return src_tok, trg_tok

    def process_data(self, data):
        src, trg = data
        src_tok = self.src_tokenizer(src.lower())
        trg_tok = self.trg_tokenizer(trg.lower())
        # print(src_tok)
        # print(trg_tok)

        src_emb = [BOS_IDX] + [self.vocab['src'][t.lower()]
                               for t in src_tok] + [EOS_IDX]
        trg_emb = [BOS_IDX] + [self.vocab['trg'][t.lower()]
                               for t in trg_tok] + [EOS_IDX]

        return src_emb, trg_emb

    def collate_fn(self, batch):
        src_batch, trg_batch = [], []
        for sample in batch:
            src_sample, trg_sample = self.process_data(sample)
            src_batch.append(torch.tensor(src_sample))
            trg_batch.append(torch.tensor(trg_sample))
        src_batch = pad_sequence(
            src_batch, padding_value=PAD_IDX, batch_first=True)
        trg_batch = pad_sequence(
            trg_batch, padding_value=PAD_IDX, batch_first=True)
        return src_batch, trg_batch

    def __len__(self):
        return self.data_size // self.batch_size