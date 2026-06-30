"""
data_loader.py — Kinyarwanda tokenizer wrapper + GPT dataset/dataloader.

The dataset/sliding-window logic is from the book (chapter 2). The only change
is that the tokenizer is our Kinyarwanda BPE instead of tiktoken's English GPT-2
encoding.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

EOT = "<|endoftext|>"


class KinyaTokenizer:
    def __init__(self, path="tokenizer/kinyarwanda_bpe/tokenizer.json"):
        self.tok = Tokenizer.from_file(path)
        self.eot_id = self.tok.token_to_id(EOT)

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, ids):
        return self.tok.decode(ids)

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids, self.target_ids = [], []
        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(torch.tensor(token_ids[i:i + max_length]))
            self.target_ids.append(torch.tensor(token_ids[i + 1:i + max_length + 1]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(text, tokenizer, batch_size=16, max_length=256,
                      stride=256, shuffle=True, drop_last=True):
    ds = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=0)
