import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer

import pickle
from helper import *


class TransformerDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.dataset = self.load_data()
        self.src = self.dataset['setting'].src_lang
        self.trg = self.dataset['setting'].trg_lang
        self.vocab = self.dataset['vocab']
        self.data = self.dataset[split]
        self.src_tokenizer, self.trg_tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        src_tok = get_tokenizer('spacy', language=LANG_MODEL[self.src])
        trg_tok = get_tokenizer('spacy', language=LANG_MODEL[self.trg])
        return src_tok, trg_tok

    def load_data(self):
        with open(self.data_dir, 'rb') as f:
            return pickle.load(f)
        return

    def process_data(self, data):
        trg, src = data
        src_tok = self.src_tokenizer(src)
        trg_tok = self.trg_tokenizer(trg)

        src_emb = [BOS_IDX] + [self.vocab['src'][t]
                               for t in src_tok] + [EOS_IDX]
        trg_emb = [BOS_IDX] + [self.vocab['trg'][t]
                               for t in trg_tok] + [EOS_IDX]

        return src_emb, trg_emb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.process_data(self.data[index])
        return data


class TransformerDataLoader(DataLoader):
    def __init__(self, data_dir, split, batch_size, num_workers, shuffle=True):
        dataset = TransformerDataset(data_dir, split=split)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, collate_fn=self.collate_fn)
        return

    def collate_fn(self, batch):
        src_batch, trg_batch = [], []
        for src_sample, trg_sample in batch:
            src_batch.append(torch.tensor(src_sample))
            trg_batch.append(torch.tensor(trg_sample))
        src_batch = pad_sequence(
            src_batch, padding_value=PAD_IDX, batch_first=True)
        trg_batch = pad_sequence(
            trg_batch, padding_value=PAD_IDX, batch_first=True)
        return src_batch, trg_batch


if __name__ == "__main__":
    data_path = "./wmttest_en_de_processed_bpe.pkl"
    test_loader = TransformerDataLoader(data_path, 'test', batch_size=5)
    src, trg = next(iter(test_loader))
    print("Source shape: ", src.shape)
    print("Target shape: ", trg.shape)
