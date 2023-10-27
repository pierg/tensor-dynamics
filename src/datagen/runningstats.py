import numpy as np

from src.shared.utils import format_section
import tensorflow as tf
import numpy as np
from typing import Union


class RunningStats:
    """
    Maintains running statistics including mean, variance, min, max,
    shape, data type, and an example of the data points.
    """

    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None
        self.min = None
        self.max = None
        self.shape = None  # New attribute to store the shape of the data
        self.dtype = None  # New attribute to store the data type
        self.example = None  # New attribute to store an example data point

    def clear(self):
        self.n = 0
        self.mean = None
        self.M2 = None
        self.min = None
        self.max = None
        self.shape = None
        self.dtype = None
        self.example = None

    def push(self, x):
        # Store the shape and data type of the first data point
        if self.n == 0:
            self.shape = x.shape
            self.dtype = str(x.dtype)
            self.example = x  # Keep the first data point as an example

            self.mean = np.zeros_like(x)
            self.M2 = np.zeros_like(x)
            self.min = np.full_like(x, np.inf)
            self.max = np.full_like(x, -np.inf)

        self.n += 1

        # Update mean and M2 for variance calculation
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

        # Update min and max
        self.min = np.minimum(self.min, x)
        self.max = np.maximum(self.max, x)

    def get_mean(self):
        return self.mean if self.mean is not None else 0

    def get_variance(self):
        return (
            self.M2 / (self.n - 1)
            if self.n > 1
            else (self.M2 if self.M2 is not None else 0)
        )

    def get_standard_deviation(self):
        return np.sqrt(self.get_variance())

    def get_min(self):
        return self.min if self.min is not None else 0

    def get_max(self):
        return self.max if self.max is not None else 0

    def get_averages(self) -> dict:
        """
        Calculate the averages of the statistical properties, treating the data as if it were flat.
        These averages serve as a summary of the data characteristics over all the samples seen so far.
        """
        # Compute the averages for each statistical property across all elements.
        averages = {
            "shape": self.shape,
            "type": self.dtype,
            "average_mean": np.mean(self.mean) if self.mean is not None else None,
            "average_variance": np.mean(self.M2 / (self.n - 1))
            if self.n > 1 and self.M2 is not None
            else None,  # Using variance formula here
            "average_std_dev": np.mean(np.sqrt(self.M2 / (self.n - 1)))
            if self.n > 1 and self.M2 is not None
            else None,  # Using std dev formula here
            "average_min": np.mean(self.min) if self.min is not None else None,
            "average_max": np.mean(self.max) if self.max is not None else None,
        }

        return averages

    def to_dict(self) -> dict:
        """
        Convert the RunningStats object to a dictionary format that is JSON serializable.
        The dictionary contains averages as summary statistics and complete statistics for detailed analysis.
        """
        # Helper function to convert numpy objects to native Python types for JSON serialization
        serialize_array = lambda x: x.tolist() if isinstance(x, np.ndarray) else x

        averages = self.get_averages()

        # Construct a dictionary with hierarchical structure: averages followed by complete stats.
        stats_dict = {
            "averages": {
                key: serialize_array(value) for key, value in averages.items()
            },
            "full": {
                "mean": serialize_array(self.get_mean()),
                "variance": serialize_array(self.get_variance()),
                "std_dev": serialize_array(self.get_standard_deviation()),
                "min": serialize_array(self.get_min()),
                "max": serialize_array(self.get_max()),
                "shape": self.shape,  # Assuming shape is a tuple, which is JSON serializable.
                "dtype": str(
                    self.dtype
                ),  # JSON serialization: convert explicitly to string
                "example": serialize_array(self.example),  # Convert example data point
            },
        }

        return stats_dict


class RunningStatsDatapoints:
    """
    Maintains running statistics for a dataset, including features and labels.
    """

    def __init__(self):
        self.features = RunningStats()
        self.labels = RunningStats()
        self.n = 0

    def update(self, feature, label):
        self.features.push(feature)
        self.labels.push(label)
        self.n += 1

    def get_feature_stats(self):
        return self.features.to_dict()

    def get_label_stats(self):
        return self.labels.to_dict()

    def __str__(self) -> str:
        """
        Create a formatted string of the averages of statistics for features and labels.
        This can be used for display or logging, giving a quick overview of the data's general characteristics.
        """
        feature_averages = self.features.get_averages()
        label_averages = self.labels.get_averages()

        # Creating a formatted string for features and labels using the averages.
        features_str = (
            "Features Averages:\n"
            + "\n".join(f"{key}: {value}" for key, value in feature_averages.items())
            + "\n"
        )
        labels_str = "Labels Averages:\n" + "\n".join(
            f"{key}: {value}" for key, value in label_averages.items()
        )

        return features_str + "\n" + labels_str

    def to_dict(self, reduced: bool = False) -> dict:
        """
        Create a dictionary representation of statistics for features and labels.

        Returns:
            A dictionary containing the statistics of features and labels.
        """

        if reduced:
            return {
                "samples": self.n,
                "features": self.features.get_averages(),
                "labels": self.labels.get_averages(),
            }

        return {
            "samples": self.n,
            "features": self.get_feature_stats(),
            "labels": self.get_label_stats(),
        }


def calculate_dataset_running_stats(
    dataset: tf.data.Dataset,
    feature_dims: Union[int, tuple] = None,
    label_dims: Union[int, tuple] = None,
) -> RunningStatsDatapoints:
    """
    Given a tf.data.Dataset, iterates through the dataset to calculate running statistics
    for both features and labels using the RunningStatsDatapoints class.

    Args:
    dataset (tf.data.Dataset): A batched or unbatched dataset of (feature, label) pairs.
    feature_dims (Union[int, tuple]): The number of dimensions or shape of the individual feature.
                                      Needed for reshaping the tensor for running stats update.
    label_dims (Union[int, tuple]): The number of dimensions or shape of the individual label.
                                    Needed for reshaping the tensor for running stats update.

    Returns:
    RunningStatsDatapoints: An object containing the running statistics for the features and labels.
    """

    running_stats_datapoints = RunningStatsDatapoints()

    # Iterate over each batch of data in the dataset
    for features, labels in dataset:
        # Convert tensors to numpy for compatibility with RunningStats methods
        features_numpy = features.numpy()
        labels_numpy = labels.numpy()

        # If the dataset is batched, we iterate through the batch.
        if feature_dims is not None and label_dims is not None:
            batch_size = features_numpy.shape[0]
            for i in range(batch_size):
                # Reshape if required and update stats for individual datapoint
                feature = (
                    features_numpy[i].reshape(feature_dims)
                    if feature_dims
                    else features_numpy[i]
                )
                label = (
                    labels_numpy[i].reshape(label_dims)
                    if label_dims
                    else labels_numpy[i]
                )
                running_stats_datapoints.update(feature, label)
        else:
            # If the dataset is unbatched, we directly update stats
            running_stats_datapoints.update(features_numpy, labels_numpy)

    return running_stats_datapoints
