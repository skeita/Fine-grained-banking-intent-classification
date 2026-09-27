"""Educational PyTorch BiLSTM for BANKING77."""

from collections import Counter

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def build_vocab(texts, min_freq=1):
    """Build a training-only vocabulary with PAD and UNK tokens."""
    counts = Counter(token for text in texts for token in text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in sorted(counts.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text, vocab):
    """Convert one query to token IDs, retaining an UNK fallback."""
    ids = [vocab.get(token, vocab["<UNK>"]) for token in text.lower().split()]
    return ids or [vocab["<UNK>"]]


def collate_batch(batch, vocab):
    """Pad a batch of (text, label) examples and return lengths."""
    encoded = [encode_text(text, vocab) for text, _ in batch]
    lengths = torch.tensor([len(x) for x in encoded], dtype=torch.long)
    max_len = int(lengths.max())
    inputs = torch.zeros((len(encoded), max_len), dtype=torch.long)
    for i, ids in enumerate(encoded):
        inputs[i, :len(ids)] = torch.tensor(ids)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    return inputs, lengths, labels


class BiLSTMClassifier(nn.Module):
    """Embedding -> bidirectional LSTM -> dropout -> 77-class logits."""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_classes=77, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(representation))
