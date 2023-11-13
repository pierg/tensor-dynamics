from dataclasses import dataclass
from typing import Optional

import tensorflow as tf

from dataset.statistics_generator import RunningStatsDatapoints

def dataset_size(dataset: tf.data.Dataset) -> int:
    """
    Calculate the number of samples in a TensorFlow dataset.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset to calculate the size for.

    Returns:
        int: The number of samples in the dataset.
    """
    return sum(1 for _ in dataset)

@dataclass
class Datasets:
    """
    Data class representing training, validation, and test datasets along with their statistics.

    Attributes:
        train_dataset (tf.data.Dataset): The training dataset.
        validation_dataset (tf.data.Dataset): The validation dataset.
        test_dataset (tf.data.Dataset): The test dataset.
        train_stats (Optional[RunningStatsDatapoints]): Running statistics for the training dataset.
        validation_stats (Optional[RunningStatsDatapoints]): Running statistics for the validation dataset.
        test_stats (Optional[RunningStatsDatapoints]): Running statistics for the test dataset.
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
        Create a Datasets instance from dictionaries containing datasets and their statistics.

        Args:
            datasets (dict): Dictionary containing TensorFlow datasets for training, validation, and testing.
            stats (dict): Dictionary containing statistics for the corresponding datasets.

        Returns:
            Datasets: An instance of the Datasets class populated with the provided datasets and statistics.
        """
        # Validate the presence of required keys in the input dictionaries
        required_keys = ["training", "validation", "testing"]
        for key in required_keys:
            if key not in datasets or key not in stats:
                raise ValueError(f"The '{key}' key is missing in the provided dictionaries.")

        # Construct and return the Datasets instance
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
        Convert the Datasets instance to a dictionary format for easy access and manipulation.

        Returns:
            dict: Dictionary representation of the Datasets instance.
        """
        datasets_dict = {
            "samples": {
                "training": dataset_size(self.train_dataset),
                "validation": dataset_size(self.validation_dataset),
                "test": dataset_size(self.test_dataset),
            }
        }

        if self.train_stats:
            datasets_dict["training_stats"] = self.train_stats.to_dict(reduced=True)
        if self.validation_stats:
            datasets_dict["validation_stats"] = self.validation_stats.to_dict(reduced=True)
        if self.test_stats:
            datasets_dict["test_stats"] = self.test_stats.to_dict(reduced=True)

        return datasets_dict

def print_dataset_statistics(datasets: Datasets) -> None:
    """
    Print basic statistics of the given datasets.

    Args:
        datasets (Datasets): The Datasets instance containing the datasets to print statistics for.
    """
    print("\nDataset Statistics:")
    print(f"Training samples: {dataset_size(datasets.train_dataset)}")
    print(f"Validation samples: {dataset_size(datasets.validation_dataset)}")
    print(f"Test samples: {dataset_size(datasets.test_dataset)}")
    if datasets.train_stats:
        print(f"Training stats: {datasets.train_stats}")
    if datasets.validation_stats:
        print(f"Validation stats: {datasets.validation_stats}")
    if datasets.test_stats:
        print(f"Test stats: {datasets.test_stats}")
