from dataclasses import dataclass, field
import tensorflow as tf
from datagen.runningstats import RunningStatsDatapoints

from dataclasses import dataclass, field
import tensorflow as tf
from typing import Optional

@dataclass
class Datasets:
    train_dataset: tf.data.Dataset
    validation_dataset: tf.data.Dataset
    test_dataset: tf.data.Dataset
    train_stats: Optional[RunningStatsDatapoints] = None
    validation_stats: Optional[RunningStatsDatapoints] = None
    test_stats: Optional[RunningStatsDatapoints] = None

    @classmethod
    def from_dict(cls, datasets: dict, stats: dict) -> "Datasets":
        # Ensure the input dictionaries have the correct keys
        if not all(key in datasets for key in ["training", "validation", "testing"]):
            raise ValueError(
                "The 'datasets' dictionary must contain 'training', 'validation', and 'testing' keys."
            )

        if not all(key in stats for key in ["training", "validation", "testing"]):
            raise ValueError(
                "The 'stats' dictionary must contain 'training', 'validation', and 'testing' keys."
            )

        # Create the Datasets instance using the provided dictionaries
        return cls(
            train_dataset=datasets["training"],
            validation_dataset=datasets["validation"],
            test_dataset=datasets["testing"],
            train_stats=stats["training"],
            validation_stats=stats["validation"],
            test_stats=stats["testing"],
        )

    def to_dict(self) -> dict:
        datasets_dict = {
            "train": {"stats": self.train_stats.to_dict(reduced=True) if self.train_stats else None},
            "validation": {"stats": self.validation_stats.to_dict(reduced=True) if self.validation_stats else None},
            "test": {"stats": self.test_stats.to_dict(reduced=True) if self.test_stats else None},
        }

        return datasets_dict


