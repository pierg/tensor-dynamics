from dataclasses import dataclass, field
import tensorflow as tf
from src.datagen.runningstats import RunningStatsDatapoints


@dataclass
class Datasets:
    train_dataset: tf.data.Dataset
    validation_dataset: tf.data.Dataset
    test_dataset: tf.data.Dataset
    train_stats: RunningStatsDatapoints
    validation_stats: RunningStatsDatapoints
    test_stats: RunningStatsDatapoints

    @classmethod
    def from_dict(cls, datasets: dict, stats: dict) -> 'Datasets':
        """
        Create a Datasets instance from provided dictionaries.

        Args:
            datasets (dict): A dictionary containing TensorFlow datasets for training, validation, and testing.
            stats (dict): A dictionary containing statistics objects for training, validation, and testing datasets.

        Returns:
            Datasets: An instance of Datasets dataclass.
        """
        # Ensure the input dictionaries have the correct keys
        if not all(key in datasets for key in ["training", "validation", "testing"]):
            raise ValueError("The 'datasets' dictionary must contain 'training', 'validation', and 'testing' keys.")

        if not all(key in stats for key in ["training", "validation", "testing"]):
            raise ValueError("The 'stats' dictionary must contain 'training', 'validation', and 'testing' keys.")

        # Create the Datasets instance using the provided dictionaries
        return cls(
            train_dataset=datasets["training"],
            validation_dataset=datasets["validation"],
            test_dataset=datasets["testing"],
            train_stats=stats["training"],
            validation_stats=stats["validation"],
            test_stats=stats["testing"]
        )


    def to_dict(self) -> dict:
        """
        Convert the Datasets instance into a dictionary containing key statistics and dataset sizes.

        Returns:
            dict: A dictionary representation of the Datasets instance.
        """
        datasets_dict = {
            "train": {
                "stats": self.train_stats.to_dict(reduced=True)
            },
            "validation": {
                "stats": self.validation_stats.to_dict(reduced=True)
            },
            "test": {
                "stats": self.test_stats.to_dict(reduced=True)
            }
        }

        return datasets_dict