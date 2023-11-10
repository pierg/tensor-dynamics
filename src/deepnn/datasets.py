from dataclasses import dataclass
import tensorflow as tf
from typing import Optional
from dataset.statistics_generator import RunningStatsDatapoints



def dataset_size(dataset: tf.data.Dataset) -> int:
    """
    Calculate the number of samples in a given dataset.

    Args:
    - dataset (tf.data.Dataset): The dataset to calculate the size for.

    Returns:
    - int: The size of the dataset.
    """
    return sum(1 for _ in dataset)

@dataclass
class Datasets:
    """
    Data class to represent datasets and their corresponding statistics.
    """

    train_dataset: tf.data.Dataset
    validation_dataset: tf.data.Dataset
    test_dataset: tf.data.Dataset
    train_stats: Optional[RunningStatsDatapoints] = None
    validation_stats: Optional[RunningStatsDatapoints] = None
    test_stats: Optional[RunningStatsDatapoints] = None

    @classmethod
    def from_dict(cls, datasets: dict, stats: dict) -> "Datasets":
        """
        Create a Datasets instance using provided dictionaries.

        Args:
        - datasets (dict): Dictionary containing train, validation, and test datasets.
        - stats (dict): Dictionary containing train, validation, and test statistics.

        Returns:
        - Datasets: An instance of the Datasets class.
        """
        # Ensure the input dictionaries have the correct keys
        required_keys = ["training", "validation", "testing"]

        if not all(key in datasets for key in required_keys):
            raise ValueError(
                "The 'datasets' dictionary must contain 'training', 'validation', and 'testing' keys."
            )

        if not all(key in stats for key in required_keys):
            raise ValueError(
                "The 'stats' dictionary must contain 'training', 'validation', and 'testing' keys."
            )

        return cls(
            train_dataset=datasets["training"],
            validation_dataset=datasets["validation"],
            test_dataset=datasets["testing"],
            train_stats=stats["training"],
            validation_stats=stats["validation"],
            test_stats=stats["testing"],
        )

    def to_dict(self) -> dict:
        """
        Convert the Datasets instance to a dictionary format.

        Returns:
        - dict: A dictionary representation of the Datasets instance.
        """
        if self.validation_stats is None and self.test_stats is None:
            return {
            "samples": {
                "training": dataset_size(self.train_dataset),
                "validation": dataset_size(self.validation_dataset),
                "test": dataset_size(self.test_dataset),
            },
                "training_stats":self.train_stats.to_dict(reduced=True)
            }
        return {
            "train": {
                "stats": self.train_stats.to_dict(reduced=True)
                if self.train_stats
                else None
            },
            "validation": {
                "stats": self.validation_stats.to_dict(reduced=True)
                if self.validation_stats
                else None
            },
            "test": {
                "stats": self.test_stats.to_dict(reduced=True)
                if self.test_stats
                else None
            },
        }


def print_dataset_statistics(datasets: Datasets) -> None:
    """
    Print statistics of the datasets.

    Args:
        datasets (Datasets): The datasets object containing train, validation, and test datasets.
    """
    print("\nDataset Statistics:")
    print(f"Training samples: {len(list(datasets.train_dataset))}")
    print(f"Validation samples: {len(list(datasets.validation_dataset))}")
    print(f"Test samples: {len(list(datasets.test_dataset))}")
    print(f"Training stats: {datasets.train_stats}\n")
