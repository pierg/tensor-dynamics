from click import Path
import numpy as np
import tensorflow as tf
import toml
from datagen.michaelis import generate_datapoint, generate_parameters
from datagen.runningstats import RunningStatsDatapoints
from shared import (
    data_config_file, data_statistics_folder
)
import hashlib
import pickle
import os
# Constants
HASH_LENGTH = 5


HASH_LENGTH = 5

class DatasetGenerator:
    def __init__(self, dataset_config, dataset_parameters, data_statistics_folder: Path = data_statistics_folder):
        self.set_initial_attributes(dataset_config, dataset_parameters, data_statistics_folder)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.ensure_statistics_available()

    def set_initial_attributes(self, dataset_config, dataset_parameters, data_statistics_folder: Path):
        """Initialize class attributes using provided arguments."""
        self.seed = dataset_config["seed"]
        self.n_samples = dataset_config["n_samples"]
        self.batch_size = dataset_config["batch_size"]
        self.shuffle_buffer_size = dataset_config.get("shuffle_buffer_size", 0)
        self.splits = dataset_config["splits"]
        self.parameters = dataset_parameters
        self.running_stats = RunningStatsDatapoints()
        self.stats_file_path = self.calculate_stats_file_path(dataset_config, dataset_parameters, data_statistics_folder)

    def calculate_stats_file_path(self, dataset_config, dataset_parameters, data_statistics_folder: Path) -> Path:
        """Calculate the statistics file path using MD5 hashing."""
        combined_data = str(dataset_config) + str(dataset_parameters)
        hash_value = hashlib.md5(combined_data.encode()).hexdigest()[:HASH_LENGTH]
        return data_statistics_folder / f"running_stats_{hash_value}.pkl"

    def ensure_statistics_available(self):
        """Ensure that statistics are either loaded or calculated."""
        if os.path.exists(self.stats_file_path):
            self.load_statistics()
        else:
            self.calculate_and_save_statistics()

    def load_statistics(self):
        """Load running statistics from file."""
        print("Loading statistics...")
        with open(self.stats_file_path, 'rb') as file:
            self.running_stats = pickle.load(file)

    def calculate_and_save_statistics(self):
        """Calculate and save running statistics to file."""
        print("Calculating statistics...")
        for _ in self._data_generator():
            pass
        with open(self.stats_file_path, 'wb') as file:
            pickle.dump(self.running_stats, file)

    def _data_generator(self):
        """A generator that yields batches of data."""
        total_batches = self.n_samples // self.batch_size
        print_every_n_batches = 10
        print()
        for batch_num in range(total_batches):
            x_batch = [generate_datapoint(generate_parameters(self.parameters))[0] for _ in range(self.batch_size)]
            y_batch = [generate_datapoint(generate_parameters(self.parameters))[1] for _ in range(self.batch_size)]
            x_batch = np.stack(x_batch)
            y_batch = np.stack(y_batch)
            self.running_stats.update(x_batch, y_batch)

            # Print a message every n batches
            if (batch_num + 1) % print_every_n_batches == 0:
                print(f"Generated {batch_num + 1} batches out of {total_batches}")

            yield x_batch, y_batch

    def create_tf_datasets(self):
        """Create TensorFlow datasets for training, validation, and testing."""
        def generator():
            """A generator that yields normalized batches of data."""
            for x, y in self._data_generator():
                normalized_x = (x - self.running_stats.features.get_mean()) / self.running_stats.features.get_standard_deviation()
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
                tf.TensorSpec(shape=(self.batch_size, *first_batch_x.shape[1:]), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size, *first_batch_y.shape[1:]), dtype=tf.float32)
            )
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


def print_first_n_elements(dataset, n=5):
    for i, (x, y) in enumerate(dataset.take(n)):
        print(f"--- Sample {i+1} ---")
        print("Features:")
        print(np.array2string(x.numpy(), precision=2, suppress_small=True))
        print("\nLabels:")
        print(np.array2string(y.numpy(), precision=2, suppress_small=True))
        print("--------------------\n")

def size_testing():
# Load the dataset configuration
    dataset_config = toml.load(data_config_file)["dtest"]

    # Create the first DatasetGenerator instance and generate datasets
    dataset_generator1 = DatasetGenerator(dataset_config["dataset"], dataset_config["parameters"])
    train_dataset1, val_dataset1, test_dataset1 = dataset_generator1.create_tf_datasets()


def testing():
    # Load the dataset configuration
    dataset_config = toml.load(data_config_file)["dtest"]

    # Create the first DatasetGenerator instance and generate datasets
    dataset_generator1 = DatasetGenerator(dataset_config["dataset"], dataset_config["parameters"])
    train_dataset1, val_dataset1, test_dataset1 = dataset_generator1.create_tf_datasets()

    # Create the second DatasetGenerator instance and generate datasets
    dataset_generator2 = DatasetGenerator(dataset_config["dataset"], dataset_config["parameters"])
    train_dataset2, val_dataset2, test_dataset2 = dataset_generator2.create_tf_datasets()

    # Check if the datasets are the same between the two DatasetGenerator instances (based on basic statistics)
    def compare_datasets(dataset1_func, dataset2_func):
        dataset1 = dataset1_func()
        dataset2 = dataset2_func()
        
        sample1 = next(iter(dataset1))
        sample2 = next(iter(dataset2))
        
        return np.allclose(sample1[0], sample2[0]) and np.allclose(sample1[1], sample2[1])

    # Using lambda functions to recreate datasets for comparison
    are_datasets_same = all([
        compare_datasets(lambda: dataset_generator1.create_tf_datasets()[0], lambda: dataset_generator2.create_tf_datasets()[0]),  # train datasets
        compare_datasets(lambda: dataset_generator1.create_tf_datasets()[1], lambda: dataset_generator2.create_tf_datasets()[1]),  # val datasets
        compare_datasets(lambda: dataset_generator1.create_tf_datasets()[2], lambda: dataset_generator2.create_tf_datasets()[2])   # test datasets
    ])

    print("Datasets recreated consistently:", are_datasets_same)

    dataset_generator1.print_statistics()
    print_first_n_elements(train_dataset1, 2)


    dataset_generator2.print_statistics()
    print_first_n_elements(train_dataset2, 2)


size_testing()