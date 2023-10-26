import numpy as np


class RunningStats:
    """
    Maintains running statistics including mean, variance, min, and max
    without storing all data points.
    """

    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None
        self.min = None
        self.max = None

    def clear(self):
        self.n = 0
        self.mean = None
        self.M2 = None
        self.min = None
        self.max = None

    def push(self, x):
        if self.n == 0:
            self.mean = np.zeros_like(x)
            self.M2 = np.zeros_like(x)
            self.min = np.full_like(
                x, np.inf
            )  # Set to infinity to ensure any value will be smaller
            self.max = np.full_like(
                x, -np.inf
            )  # Set to negative infinity to ensure any value will be larger

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


class RunningStatsDatapoints:
    """
    Maintains running statistics for a dataset, including features and labels.
    """

    def __init__(self):
        self.features = RunningStats()
        self.labels = RunningStats()

    def update(self, feature, label):
        self.features.push(feature)
        self.labels.push(label)

    def get_feature_stats(self):
        return {
            "mean": self.features.get_mean(),
            "variance": self.features.get_variance(),
            "std_dev": self.features.get_standard_deviation(),
            "min": self.features.get_min(),
            "max": self.features.get_max(),
        }

    def get_label_stats(self):
        return {
            "mean": self.labels.get_mean(),
            "variance": self.labels.get_variance(),
            "std_dev": self.labels.get_standard_deviation(),
            "min": self.labels.get_min(),
            "max": self.labels.get_max(),
        }
