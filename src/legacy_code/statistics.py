'''
Author: Piergiuseppe Mallozzi
Date: November 2023
'''

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import toml

from dataset.michaelis import generate_datapoint, generate_parameters


def plot_and_save_statistics(cumulative_means, title, file_name):
    """
    Plots the cumulative means and saves the plot to a file.

    :param cumulative_means: List of cumulative means.
    :param title: The title of the plot.
    :param file_name: The name of the file to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_means, label="Average Cumulative Mean")
    plt.xlabel("Number of Samples")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.savefig(file_name)
    plt.close()

    ...


class RunningStats:
    """
    Maintains running statistics including mean and standard deviation without storing all data points.
    """

    def __init__(self, shape):
        self.n = 0
        self.old_mean = np.zeros(shape)
        self.new_mean = np.zeros(shape)
        self.old_std = np.zeros(shape)
        self.new_std = np.zeros(shape)

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_mean = self.new_mean = x
            self.old_std = np.zeros(x.shape)
        else:
            self.new_mean = self.old_mean + (x - self.old_mean) / self.n
            self.new_std = self.old_std + (x - self.old_mean) * (x - self.new_mean)

            self.old_mean = self.new_mean
            self.old_std = self.new_std

    def mean(self):
        return self.new_mean if self.n else self.old_mean

    def variance(self):
        return self.new_std / (self.n - 1) if self.n > 1 else self.old_std

    def standard_deviation(self):
        return np.sqrt(self.variance())


class RunningStats:
    """
    Maintains running statistics including mean and variance without storing all data points.
    """

    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None

    def clear(self):
        self.n = 0

    def push(self, x):
        if self.n == 0:
            self.mean = np.zeros_like(x)
            self.M2 = np.zeros_like(x)

        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean if self.mean is not None else 0

    def get_variance(self):
        # For sample variance, use M2 / (n - 1)
        # For population variance, use M2 / n
        return (
            self.M2 / (self.n - 1)
            if self.n > 1
            else (self.M2 if self.M2 is not None else 0)
        )

    def get_standard_deviation(self):
        return np.sqrt(self.get_variance())


class RunningStatsDataset:
    """
    Maintains running statistics for a dataset, including features and labels.
    """

    def __init__(self):
        self.features = RunningStats()
        self.labels = RunningStats()
        self.n = 0
        self.initialized = False

    def update(self, feature, label):
        # If this is the first update, we don't need to do anything special for initialization,
        # because our RunningStats objects now initialize themselves on the first push.
        self.features.push(feature)
        self.labels.push(label)
        self.n += 1

    def get_feature_stats(self):
        return {
            "mean": self.features.get_mean(),
            "variance": self.features.get_variance(),
            "std_dev": self.features.get_standard_deviation(),
        }

    def get_label_stats(self):
        return {
            "mean": self.labels.get_mean(),
            "variance": self.labels.get_variance(),
            "std_dev": self.labels.get_standard_deviation(),
        }


def compute_mean_std(data_config_file: Path, data_config_id: str = "d1"):
    # Load the dataset configuration from the TOML file
    data_config = toml.load(data_config_file)
    dataset_config = data_config[data_config_id]

    num_samples = dataset_config["dataset"]["n_samples"]
    param_config = dataset_config["parameters"]

    # Initialize RunningStats instances for calculating statistics
    # We're initializing with zero-size dimensions as we don't know the exact shape until we generate data points.
    stats_features = RunningStats(shape=(0,))
    stats_labels = RunningStats(shape=(0,))

    cumulative_means_features = []
    cumulative_means_labels = []

    for _ in range(num_samples):
        parameters = generate_parameters(param_config)
        features, labels = generate_datapoint(parameters)
        features_flat = features.flatten()

        # Check if our RunningStats instances are uninitialized (zero-size)
        if stats_features.n == 0:
            stats_features = RunningStats(shape=features_flat.shape)
            stats_labels = RunningStats(shape=labels.shape)

        # Push the current samples for statistics calculation
        stats_features.push(features_flat)
        stats_labels.push(labels)

        # For the graph, we're only interested in the mean of the features across all samples
        cumulative_means_features.append(np.mean(stats_features.mean()))
        cumulative_means_labels.append(np.mean(stats_labels.mean()))

    # Retrieve the calculated statistics
    most_accurate_mean_features = stats_features.mean()
    most_accurate_std_features = stats_features.standard_deviation()
    most_accurate_mean_labels = stats_labels.mean()
    most_accurate_std_labels = stats_labels.standard_deviation()

    statistics_configs = {data_config_id: {}}
    # Update the dataset configuration with the new statistics
    statistics_configs[data_config_id]["statistics"] = {
        "most_accurate_mean_features": most_accurate_mean_features.tolist(),  # Convert np.array to list
        "most_accurate_std_features": most_accurate_std_features.tolist(),
        "most_accurate_mean_labels": most_accurate_mean_labels.tolist(),
        "most_accurate_std_labels": most_accurate_std_labels.tolist(),
    }

    # Save the updated configuration back to the TOML file
    statistic_file = data_config_file.parent / "statistics.toml"
    with open(statistic_file, "w") as toml_file:
        toml.dump(statistics_configs, toml_file)

    print(f"Most accurate mean (features): {most_accurate_mean_features}")
    print(f"Most accurate standard deviation (features): {most_accurate_std_features}")
    print(f"Most accurate mean (labels): {most_accurate_mean_labels}")
    print(f"Most accurate standard deviation (labels): {most_accurate_std_labels}")

    # Plot statistics for features
    plot_and_save_statistics(
        cumulative_means_features,
        title="Cumulative Statistics for Features",
        file_name=current_dir / "cumulative_statistics_features.png",
    )

    # Plot statistics for labels
    plot_and_save_statistics(
        cumulative_means_labels,
        title="Cumulative Statistics for Labels",
        file_name=current_dir / "cumulative_statistics_labels.png",
    )

    return (
        most_accurate_mean_features,
        most_accurate_std_features,
        most_accurate_mean_labels,
        most_accurate_std_labels,
    )


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    data_config_file = current_dir / "data.toml"
    data_config = toml.load(data_config_file)

    compute_mean_std(data_config_file, "d1")
