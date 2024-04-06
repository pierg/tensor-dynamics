"""
Author: Piergiuseppe Mallozzi
Date: November 2023
"""

import hashlib
import os
import pickle

import numpy as np
import tensorflow as tf
from click import Path

from dataset.michaelis import generate_datapoint, generate_parameters
from dataset.statistics_generator import RunningStatsDatapoints
from shared import data_statistics_folder

# Constants
HASH_LENGTH = 5
DTYPE = np.float32
TF_DTYPE = tf.float32


class DatasetGenerator:
    def __init__(
        self,
        dataset_config,
        dataset_parameters,
        use_stats_of: int | None,
        data_statistics_folder: Path = data_statistics_folder,
    ):
        """
        Initialize the DatasetGenerator instance.

        Args:
            dataset_config: Configuration for the dataset.
            dataset_parameters: Parameters specific to dataset generation.
            use_stats_of (int | None): Number of samples to use for statistics calculation.
            data_statistics_folder (Path): Folder to store data statistics.
        """
        if use_stats_of is None:
            use_stats_of = dataset_config["n_samples"]

        self.set_initial_attributes(
            dataset_config, dataset_parameters, use_stats_of, data_statistics_folder
        )
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def set_initial_attributes(
        self,
        dataset_config,
        dataset_parameters,
        use_stats_of,
        data_statistics_folder: Path,
    ):
        """
        Set initial attributes for the DatasetGenerator.

        Args:
            dataset_config: Dataset configuration.
            dataset_parameters: Parameters for dataset generation.
            use_stats_of: Number of samples for stats calculation.
            data_statistics_folder (Path): Path to store data statistics.
        """
        self.seed = dataset_config["seed"]
        self.n_samples = dataset_config["n_samples"]
        self.batch_size = dataset_config["batch_size"]
        self.shuffle_buffer_size = dataset_config.get("shuffle_buffer_size", 0)
        self.splits = dataset_config["splits"]
        self.parameters = dataset_parameters

        # Calculate and load or generate running statistics
        stats_file_path = self.calculate_stats_file_path(
            dataset_parameters, data_statistics_folder, use_stats_of
        )
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
        """
        Calculate the file path for storing dataset statistics.

        Args:
            dataset_parameters: Parameters for dataset generation.
            data_statistics_folder (Path): Path to store data statistics.
            use_stats_of (int): Number of samples for statistics calculation.

        Returns:
            Path: Calculated file path for statistics.
        """
        hash_value = hashlib.md5(str(dataset_parameters).encode()).hexdigest()[
            :HASH_LENGTH
        ]
        return data_statistics_folder / f"running_stats_{hash_value}_{use_stats_of}.pkl"

    def _data_generator(self):
        """
        Generator that yields batches of data points.

        Yields:
            Batches of data points (x_batch, y_batch).
        """
        total_batches = self.n_samples // self.batch_size
        print_every_n_batches = 1000
        for batch_num in range(total_batches):
            datapoints = [
                generate_datapoint(generate_parameters(self.parameters))
                for _ in range(self.batch_size)
            ]
            x_batch, y_batch = zip(*datapoints)  # Separate the tuple into x and y
            x_batch = np.stack(x_batch)
            y_batch = np.stack(y_batch)

            if (batch_num + 1) % print_every_n_batches == 0:
                print(f"Generated {batch_num + 1} batches out of {total_batches}")

            yield x_batch, y_batch

    def create_tf_datasets(self):
        """
        Create TensorFlow datasets for training, validation, and testing.

        Returns:
            Three TensorFlow datasets: train_dataset, val_dataset, test_dataset.
        """
        epsilon = 1e-7  # Prevent division by zero

        def generator():
            """Generator yielding normalized batches of data."""
            for x, y in self._data_generator():
                std_dev = self.running_stats.features.get_standard_deviation()
                normalized_x = (x - self.running_stats.features.get_mean()) / (
                    std_dev + epsilon
                )
                features = normalized_x.astype(DTYPE)  # Convert to single precision
                labels = y.astype(DTYPE)  # Convert to single precision
                features = np.round(features, 3)
                yield features, labels

        # Determine the output_signature for TensorFlow dataset
        shape_generator = self._data_generator()
        first_batch_x, first_batch_y = next(shape_generator)

        full_dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(self.batch_size, *first_batch_x.shape[1:]), dtype=TF_DTYPE
                ),
                tf.TensorSpec(
                    shape=(self.batch_size, *first_batch_y.shape[1:]), dtype=TF_DTYPE
                ),
            ),
        ).prefetch(tf.data.AUTOTUNE)

        # Split the dataset into training, validation, and testing
        train_size = int(self.splits[0] * self.n_samples / self.batch_size)
        val_size = int(self.splits[1] * self.n_samples / self.batch_size)

        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size).take(val_size)
        test_dataset = full_dataset.skip(train_size + val_size)

        return train_dataset, val_dataset, test_dataset
