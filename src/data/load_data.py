"""Helpers for loading BANKING77 without leaking test information."""

from datasets import DatasetDict, load_dataset


def load_banking77(validation_size: float = 0.1, seed: int = 42) -> DatasetDict:
    """Load BANKING77 and make a validation split from training data only.

    The original test split is returned unchanged. This matters because the
    test set should be used once, at the very end, for the final comparison.
    """
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)

    # Stratification keeps all 77 intents represented in the validation split.
    train_validation = dataset["train"].train_test_split(
        test_size=validation_size,
        seed=seed,
        stratify_by_column="label",
    )

    return DatasetDict(
        {
            "train": train_validation["train"],
            "validation": train_validation["test"],
            "test": dataset["test"],
        }
    )
