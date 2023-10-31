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

class DatasetGenerator:
    def __init__(self, 
                 dataset_config, 
                 dataset_parameters,
                 data_statistics_folder: Path = data_statistics_folder):


        self.seed = dataset_config["seed"]
        self.n_samples = dataset_config["n_samples"]
        self.batch_size = dataset_config["batch_size"]
        self.shuffle_buffer_size = dataset_config["shuffle_buffer_size"]
        self.splits = dataset_config["splits"]
        self.parameters = dataset_parameters
        self.running_stats = RunningStatsDatapoints()

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Construct a hash from dataset_config and dataset_parameters
        combined_data = str(dataset_config) + str(dataset_parameters)
        hash_value = hashlib.md5(combined_data.encode()).hexdigest()[:5]  # Taking first 5 characters
        self.stats_file_path = data_statistics_folder / f"running_stats_{hash_value}.pkl"

        if os.path.exists(self.stats_file_path):
            print("Loading statistics...")
            with open(self.stats_file_path, 'rb') as file:
                self.running_stats = pickle.load(file)
        else:
            self.running_stats = RunningStatsDatapoints()
            # Calculate running statistics
            print("Calculating statistics...")
            for _ in self._data_generator():
                pass
            # Save the computed running stats to disk
            with open(self.stats_file_path, 'wb') as file:
                pickle.dump(self.running_stats, file)

    def _data_generator(self):
        for _ in range(0, self.n_samples, self.batch_size):
            x_batch = []
            y_batch = []
            for _ in range(self.batch_size):
                parameters = generate_parameters(self.parameters)
                x, y = generate_datapoint(parameters)
                x_batch.append(x)
                y_batch.append(y)
            x_batch = np.stack(x_batch)
            y_batch = np.stack(y_batch)
            self.running_stats.update(x_batch, y_batch)
            yield x_batch, y_batch

    def create_tf_datasets(self):

        # Determine the shape by generating a single sample
        sample_generator = self._data_generator()
        sample_x, sample_y = next(sample_generator)

        def generator():
            yield sample_x, sample_y  # yield the already generated sample first
            for x, y in sample_generator:
                normalized_x = (x - self.running_stats.features.get_mean()) / self.running_stats.features.get_standard_deviation()

                # Convert to float16 for reduced memory footprint
                features = normalized_x.astype(np.float16)
                labels = y.astype(np.float16)

                # Round features to 3 decimal places
                features = np.round(features, 3)

                yield features, labels

        full_dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, *sample_x.shape[1:]), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size, *sample_y.shape[1:]), dtype=tf.float32)
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

# dataset_generator1.print_statistics()
# print_first_n_elements(train_dataset1, 2)


# dataset_generator2.print_statistics()
# print_first_n_elements(train_dataset2, 2)