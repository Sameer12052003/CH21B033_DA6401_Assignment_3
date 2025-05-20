import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb

def load_tsv_data(file_path):
    latin_words = []
    devanagari_words = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                devanagari_words.append(row[0])  # Devanagari word
                latin_words.append(row[1])       # Latin transliteration

    return latin_words, devanagari_words



def build_vocab(sequences, special_tokens=['<pad>', '<sos>', '<eos>', '<unk>']):
    vocab = set(char for seq in sequences for char in seq)
    vocab = special_tokens + sorted(vocab)

    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return char2idx, idx2char

def tokenize_sequence(seq, char2idx, sos=True, eos=True):
    tokens = [char2idx.get(c, char2idx['<unk>']) for c in seq]
    if sos:
        tokens = [char2idx['<sos>']] + tokens
    if eos:
        tokens = tokens + [char2idx['<eos>']]
    return tokens


def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = pad_sequence([torch.tensor(seq) for seq in src_seqs],
                              batch_first=True, padding_value=src_pad_idx)
    tgt_padded = pad_sequence([torch.tensor(seq) for seq in tgt_seqs],
                              batch_first=True, padding_value=tgt_pad_idx)
    return src_padded, tgt_padded


class DakshinaCharDataset(Dataset):
    def __init__(self, latin_words, devanagari_words, src_vocab, tgt_vocab):
        self.data = [
            (tokenize_sequence(latin, src_vocab),
             tokenize_sequence(dev, tgt_vocab))
            for latin, dev in zip(latin_words, devanagari_words)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def exact_match_accuracy(predictions, references):
        return sum(p == r for p, r in zip(predictions, references)) / len(predictions)

def greedy_decode(model, src, max_len,sos_token):
    model.eval()
    with torch.no_grad():
        batch_size = src.size(0)
        encoder_outputs, hidden = model.encoder(src)

        input_token = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=src.device)
        outputs = []

        for _ in range(max_len):
            output, hidden = model.decoder(input_token, hidden)
            next_token = output.argmax(-1)  # [batch_size, 1]
            outputs.append(next_token)
            input_token = next_token

        outputs = torch.cat(outputs, dim=1)  # [batch_size, max_len]
        return outputs.tolist()


