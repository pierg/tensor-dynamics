import numpy as np
from pathlib import Path
import toml
import matplotlib.pyplot as plt
from src.datagen.michaelis import generate_datapoint, generate_parameters



def plot_and_save_statistics(cumulative_means, title, file_name):
    """
    Plots the cumulative means and saves the plot to a file.

    :param cumulative_means: List of cumulative means.
    :param title: The title of the plot.
    :param file_name: The name of the file to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_means, label='Average Cumulative Mean')
    plt.xlabel('Number of Samples')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(file_name)
    plt.close()


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


def main():
    current_dir = Path(__file__).resolve().parent
    data_config_file = current_dir / "data.toml"
    data_config = toml.load(data_config_file)
    dataset_config = data_config["d1"]

    num_samples = dataset_config["dataset"]["n_samples"]
    param_config = dataset_config["parameters"]
        
    parameters = generate_parameters(param_config)
    features, labels = generate_datapoint(parameters)
    features_flat = features.flatten()
    # Each feature will be of shape (1503,) after flattening (501*3)
    # Each label is already a 1D array of shape (12,)

    stats_features = RunningStats(shape=features_flat.shape)
    stats_labels = RunningStats(shape=labels.shape)

    cumulative_means_features = []
    cumulative_means_labels = []

    for _ in range(num_samples):
        parameters = generate_parameters(param_config)
        features, labels = generate_datapoint(parameters)

        features_flat = features.flatten()
        stats_features.push(features_flat)
        stats_labels.push(labels)

        # For the graph, we're only interested in the mean of the features across all samples
        cumulative_means_features.append(np.mean(stats_features.mean()))
        cumulative_means_labels.append(np.mean(stats_labels.mean()))

    most_accurate_mean_features = stats_features.mean()
    most_accurate_std_features = stats_features.standard_deviation()
    most_accurate_mean_labels = stats_labels.mean()
    most_accurate_std_labels = stats_labels.standard_deviation()

    print(f"Most accurate mean (features): {most_accurate_mean_features}")
    print(f"Most accurate standard deviation (features): {most_accurate_std_features}")
    print(f"Most accurate mean (labels): {most_accurate_mean_labels}")
    print(f"Most accurate standard deviation (labels): {most_accurate_std_labels}")


    # Plot statistics for features
    plot_and_save_statistics(
        cumulative_means_features,
        title='Cumulative Statistics for Features',
        file_name=current_dir / 'cumulative_statistics_features.png'
    )

        # Plot statistics for labels
    plot_and_save_statistics(
        cumulative_means_labels,
        title='Cumulative Statistics for Labels',
        file_name=current_dir / 'cumulative_statistics_labels.png'
    )


    return most_accurate_mean_features, most_accurate_std_features, most_accurate_mean_labels, most_accurate_std_labels


if __name__ == "__main__":
    main()
