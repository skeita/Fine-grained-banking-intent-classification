"""A small, transparent PyTorch BiLSTM for BANKING77."""

from collections import Counter

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def build_vocab(texts, min_freq=1):
    """Build a vocabulary from the training queries.

    The vocabulary deliberately belongs to the training split only. `<PAD>`
    lets batches have a common shape, while `<UNK>` handles words that were
    not seen during training.
    """
    counts = Counter(
        token
        for text in texts
        for token in text.lower().split()
    )
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for token, count in sorted(counts.items()):
        if count >= min_freq:
            vocab[token] = len(vocab)

    return vocab


def encode_text(text, vocab):
    """Turn one customer query into the integer IDs used by the model."""
    ids = [
        vocab.get(token, vocab["<UNK>"])
        for token in text.lower().split()
    ]
    return ids or [vocab["<UNK>"]]


def collate_batch(batch, vocab):
    """Convert examples into a padded tensor batch for the DataLoader."""
    encoded = [encode_text(text, vocab) for text, _ in batch]
    lengths = torch.tensor([len(item) for item in encoded], dtype=torch.long)
    max_length = int(lengths.max())

    input_ids = torch.zeros(
        (len(encoded), max_length),
        dtype=torch.long,
    )
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids)

    labels = torch.tensor(
        [label for _, label in batch],
        dtype=torch.long,
    )
    return input_ids, lengths, labels


class BiLSTMClassifier(nn.Module):
    """Predict one of the 77 banking intents from a token sequence."""

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_classes=77,
        dropout=0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, lengths):
        """Return one logit vector for each query in the batch."""
        embedded = self.embedding(input_ids)

        # Packing prevents padding tokens from influencing the recurrent model.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)

        # The final forward and backward states form the sentence representation.
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(representation))
