"""Training helpers for the DistilBERT experiments."""

import copy
import time

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.utils import get_device, set_seed

MODEL_NAME = "distilbert/distilbert-base-uncased"


def prepare_dataset(dataset, tokenizer, max_length=64):
    """Tokenize each split while leaving the original text untouched."""
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
    )


def run_transformer_experiment(
    dataset,
    frozen=False,
    learning_rate=2e-5,
    epochs=3,
    max_length=64,
    seed=42,
):
    """Train DistilBERT and keep the checkpoint with the best validation F1."""
    set_seed(seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = prepare_dataset(dataset, tokenizer, max_length)
    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=16,
        shuffle=True,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        tokenized["validation"],
        batch_size=32,
        shuffle=False,
        collate_fn=collator,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=77,
    ).to(device)

    if frozen:
        # E2 trains only the classification head, not the encoder itself.
        for parameter in model.distilbert.parameters():
            parameter.requires_grad = False

    trainable_parameters = (
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer = AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=0.01,
    )

    best_state = None
    best_f1 = -1.0
    history = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            train_loss += output.loss.item() * batch["labels"].size(0)

        model.eval()
        validation_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in validation_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                validation_loss += output.loss.item() * batch["labels"].size(0)
                y_true.extend(batch["labels"].cpu().tolist())
                y_pred.extend(output.logits.argmax(1).cpu().tolist())

        validation_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader.dataset),
            "validation_loss": validation_loss / len(validation_loader.dataset),
            "validation_accuracy": accuracy_score(y_true, y_pred),
            "validation_macro_f1": validation_f1,
        }
        history.append(row)

        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history,
        "training_seconds": time.perf_counter() - start,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
