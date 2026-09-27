"""BANKING77 loading and reproducible train/validation splitting."""

from datasets import DatasetDict, load_dataset


def load_banking77(validation_size: float = 0.1, seed: int = 42) -> DatasetDict:
    """Load BANKING77 and split only the official training set."""
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    split = dataset["train"].train_test_split(
        test_size=validation_size, seed=seed, stratify_by_column="label"
    )
    return DatasetDict({"train": split["train"], "validation": split["test"], "test": dataset["test"]})
