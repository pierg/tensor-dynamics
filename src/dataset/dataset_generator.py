from click import Path
import numpy as np
import tensorflow as tf
import toml
from dataset.michaelis import generate_datapoint, generate_parameters
from dataset.statistics_generator import RunningStatsDatapoints
from shared import data_statistics_folder
import hashlib
import pickle
import os

# Constants
HASH_LENGTH = 5


class DatasetGenerator:
    def __init__(
        self,
        dataset_config,
        dataset_parameters,
        use_stats_of: int | None,
        data_statistics_folder: Path = data_statistics_folder,
    ):
        if use_stats_of is None:
            use_stats_of = dataset_config["n_samples"]

        self.set_initial_attributes(
            dataset_config, dataset_parameters, use_stats_of, data_statistics_folder
        )
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def set_initial_attributes(
        self, dataset_config, dataset_parameters, use_stats_of, data_statistics_folder: Path
    ):
        """Initialize class attributes using provided arguments."""
        self.seed = dataset_config["seed"]
        self.n_samples = dataset_config["n_samples"]
        self.batch_size = dataset_config["batch_size"]
        self.shuffle_buffer_size = dataset_config.get("shuffle_buffer_size", 0)
        self.splits = dataset_config["splits"]
        self.parameters = dataset_parameters
        
        # Calculate stats file path
        stats_file_path = self.calculate_stats_file_path(
            dataset_parameters, data_statistics_folder, use_stats_of
        )
        print(stats_file_path)
        exit

        # Check if stats file exists, if so load it, otherwise generate it
        if os.path.exists(stats_file_path):
            with open(stats_file_path, "rb") as f:
                print("Statistics found, loading...")
                self.running_stats = pickle.load(f)
        else:
            self.running_stats = RunningStatsDatapoints.from_generator(
                self._data_generator, file_path=stats_file_path, max_points=use_stats_of
            )

    def calculate_stats_file_path(
        self,
        dataset_parameters,
        data_statistics_folder: Path,
        use_stats_of: int,
    ) -> Path:
        """Calculate the statistics file path using MD5 hashing."""
        hash_value = hashlib.md5(str(dataset_parameters).encode()).hexdigest()[:HASH_LENGTH]
        return data_statistics_folder / f"running_stats_{hash_value}_{use_stats_of}.pkl"


    def _data_generator(self):
        """A generator that yields batches of data."""
        total_batches = self.n_samples // self.batch_size
        print_every_n_batches = 100
        for batch_num in range(total_batches):
            x_batch = [
                generate_datapoint(generate_parameters(self.parameters))[0]
                for _ in range(self.batch_size)
            ]
            y_batch = [
                generate_datapoint(generate_parameters(self.parameters))[1]
                for _ in range(self.batch_size)
            ]
            x_batch = np.stack(x_batch)
            y_batch = np.stack(y_batch)

            # Print a message every n batches
            if (batch_num + 1) % print_every_n_batches == 0:
                print(f"Generated {batch_num + 1} batches out of {total_batches}")

            yield x_batch, y_batch

    def create_tf_datasets(self):
        """Create TensorFlow datasets for training, validation, and testing."""

        def generator():
            """A generator that yields normalized batches of data."""
            for x, y in self._data_generator():
                normalized_x = (
                    x - self.running_stats.features.get_mean()
                ) / self.running_stats.features.get_standard_deviation()
                features = normalized_x.astype(np.float32)
                labels = y.astype(np.float32)
                features = np.round(features, 3)
                yield features, labels

        # Get the shape of the first batch to determine the output_signature
        shape_generator = self._data_generator()
        first_batch_x, first_batch_y = next(shape_generator)

        full_dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(self.batch_size, *first_batch_x.shape[1:]), dtype=tf.float32
                ),
                tf.TensorSpec(
                    shape=(self.batch_size, *first_batch_y.shape[1:]), dtype=tf.float32
                ),
            ),
        ).prefetch(tf.data.AUTOTUNE)

        train_size = int(self.splits[0] * self.n_samples / self.batch_size)
        val_size = int(self.splits[1] * self.n_samples / self.batch_size)

        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size).take(val_size)
        test_dataset = full_dataset.skip(train_size + val_size)

        return train_dataset, val_dataset, test_dataset

    def print_statistics(self):
        feature_stats = self.running_stats.get_feature_stats()
        label_stats = self.running_stats.get_label_stats()

        print("=== Dataset Statistics ===")
        print(f"Total samples: {self.n_samples}")
        print(f"Feature shape: {feature_stats['averages']['shape']}")
        print(f"Feature type: {feature_stats['averages']['type']}")
        print(f"Label shape: {label_stats['averages']['shape']}")
        print(f"Label type: {label_stats['averages']['type']}")
        print(f"Running stats:")
        print(str(self.running_stats))
