import numpy as np
import tensorflow as tf
import toml
import hashlib
import pickle
import os
from dataset.michaelis import generate_datapoint, generate_parameters
from src.dataset.statistics_generator import RunningStatsDatapoints
from shared import data_config_file, data_statistics_folder

# Constants
HASH_LENGTH = 5


def print_first_n_elements(dataset, n=5):
    """Print the first n samples of the dataset."""
    for i, (x, y) in enumerate(dataset.take(n)):
        print(f"--- Sample {i+1} ---")
        print("Features:")
        print(np.array2string(x.numpy(), precision=2, suppress_small=True))
        print("\nLabels:")
        print(np.array2string(y.numpy(), precision=2, suppress_small=True))
        print("--------------------\n")


def size_testing():
    """Size testing for the dataset."""
    dataset_config = toml.load(data_config_file)["dtest"]
    dataset_generator1 = DatasetGenerator(
        dataset_config["dataset"], dataset_config["parameters"]
    )
    (
        train_dataset1,
        val_dataset1,
        test_dataset1,
    ) = dataset_generator1.create_tf_datasets()


def testing():
    """Test to ensure datasets are recreated consistently."""
    dataset_config = toml.load(data_config_file)["dtest"]

    # Create two DatasetGenerator instances
    dataset_generator1 = DatasetGenerator(
        dataset_config["dataset"], dataset_config["parameters"]
    )
    dataset_generator2 = DatasetGenerator(
        dataset_config["dataset"], dataset_config["parameters"]
    )

    def compare_datasets(dataset1_func, dataset2_func):
        """Compare two datasets to ensure they are equal."""
        dataset1 = dataset1_func()
        dataset2 = dataset2_func()

        sample1 = next(iter(dataset1))
        sample2 = next(iter(dataset2))

        return np.allclose(sample1[0], sample2[0]) and np.allclose(
            sample1[1], sample2[1]
        )

    are_datasets_same = all(
        [
            compare_datasets(
                lambda: dataset_generator1.create_tf_datasets()[0],
                lambda: dataset_generator2.create_tf_datasets()[0],
            ),  # train datasets
            compare_datasets(
                lambda: dataset_generator1.create_tf_datasets()[1],
                lambda: dataset_generator2.create_tf_datasets()[1],
            ),  # val datasets
            compare_datasets(
                lambda: dataset_generator1.create_tf_datasets()[2],
                lambda: dataset_generator2.create_tf_datasets()[2],
            ),  # test datasets
        ]
    )

    print("Datasets recreated consistently:", are_datasets_same)
    dataset_generator1.print_statistics()
    print_first_n_elements(train_dataset1, 2)

    dataset_generator2.print_statistics()
    print_first_n_elements(train_dataset2, 2)
