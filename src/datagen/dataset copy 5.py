import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import toml
from typing import Tuple, List
from src.datagen.michaelis import generate_datapoint, generate_parameters, normalize_data, quantize_data
from src.datagen.runningstats import RunningStatsDatapoints
from src.deepnn.utils import compute_dataset_range, compute_mean_and_variance



def normalize_dataset(dataset: tf.data.Dataset, feature_stats: dict, label_stats: dict) -> tf.data.Dataset:
    """
    Apply normalization to the features and labels in the dataset using the provided statistics.

    Args:
        dataset (tf.data.Dataset): A batched dataset of (feature, label) pairs.
        feature_stats (dict): Dictionary containing 'mean' and 'std_dev' for features.
        label_stats (dict): Dictionary containing 'mean' and 'std_dev' for labels.

    Returns:
        tf.data.Dataset: The normalized dataset.
    """

    def normalize_fn(feature, label):
        # Normalize features and labels using the statistics provided
        normalized_feature = (feature - feature_stats['mean']) / feature_stats['std_dev']
        normalized_label = (label - label_stats['mean']) / label_stats['std_dev']

        return normalized_feature, normalized_label

    # Apply the normalization function to each element in the dataset
    normalized_dataset = dataset.map(normalize_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return normalized_dataset


def transform_and_save_dataset(original_dataset_path: Path, feature_stats: dict, label_stats: dict):
    """
    Load, normalize, and save the dataset as TFRecord.

    Args:
        original_dataset_path (Path): Path to the original TFRecord file.
        feature_stats (dict): Statistical information for features.
        label_stats (dict): Statistical information for labels.
    """
    
    # Load the dataset using existing functions
    dataset = load_dataset(original_dataset_path)

    # Normalize the dataset using the provided statistics
    normalized_dataset = normalize_dataset(dataset, feature_stats, label_stats)

    # Function to write the dataset to a TFRecord file
    def _write_dataset(dataset, filename):
        writer = tf.data.experimental.TFRecordWriter(str(filename))
        writer.write(dataset.unbatch())  # We need to unbatch to store as TFRecord

    output_dir = original_dataset_path.parent

    # Save the transformed dataset
    _write_dataset(normalized_dataset, output_dir / 'normalized_dataset.tfrecord')

    print(f"Transformed dataset saved to {output_dir / 'normalized_dataset.tfrecord'}")



def print_data_info(data: np.ndarray, label: str):
    """
    Display information about the provided data.

    Args:
        data (np.ndarray): Data to display information about.
        label (str): Label to be used in print statements (e.g. "Data" or "Predictions").
    """
    print(f"{label} type: {type(data)}")
    print(f"{label} shape: {data.shape}")
    print(f"{label} size: {data.size}")
    print(f"{label} dtype: {data.dtype}")
    print("-" * 50)




def serialize_example(feature: np.ndarray, label: np.ndarray) -> bytes:
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Feature-compatible data type.
    feature_set = {
        'feature_shape': _bytes_feature(np.array(feature.shape, dtype=np.int64)),
        'feature_data': _bytes_feature(feature.astype(np.float64)),
        'label_shape': _bytes_feature(np.array(label.shape, dtype=np.int64)),
        'label_data': _bytes_feature(label.astype(np.float64)),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_set))
    return example_proto.SerializeToString()

def parse_tfrecord_fn(example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse TFRecord format.
    """
    feature_description = {
        # Adjust these keys to match what's actually in the TFRecord.
        'feature_shape': tf.io.FixedLenFeature([], tf.string),
        'feature_data': tf.io.FixedLenFeature([], tf.string),
        'label_shape': tf.io.FixedLenFeature([], tf.string),
        'label_data': tf.io.FixedLenFeature([], tf.string),
    }

    # Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example, feature_description)
    
    # Decode the features and labels and reshape them.
    feature_shape = tf.io.parse_tensor(example['feature_shape'], out_type=tf.int64) 
    feature = tf.io.parse_tensor(example['feature_data'], out_type=tf.float64)
    feature = tf.reshape(feature, feature_shape)

    label_shape = tf.io.parse_tensor(example['label_shape'], out_type=tf.int64)
    label = tf.io.parse_tensor(example['label_data'], out_type=tf.float64)
    label = tf.reshape(label, label_shape)
    
    return feature, label



def pretty_print_dataset_elements(dataset_location: Path, n: int = None):
    """
    Pretty prints "n" elements from the dataset.

    Args:
        dataset_location (Path): The file path of the TFRecord dataset.
        n (int): The number of records to print from the dataset.
    """

    # Load and parse the dataset
    parsed_dataset = load_dataset(dataset_location, batch_size=1)

    val_range = compute_dataset_range(parsed_dataset)
    print(f"Range:\t\t{val_range}")
    mean_val, variance_val = compute_mean_and_variance(parsed_dataset)
    print(f"Mean :\t\t{mean_val}")
    print(f"Var  :\t\t{variance_val}")

    # Determine how many records to print
    dataset_to_print = parsed_dataset if n is None else parsed_dataset.take(n)

    for count, (feature, label) in enumerate(dataset_to_print, 1):
        print(f"Element: {count}")
        print("Feature:")
        print(feature.numpy())  # Assuming feature is a tensor that can be converted to a numpy array.
        print("Label:")
        print(label.numpy())  # Assuming label is a tensor that can be converted to a numpy array.
        print("-" * 30)




def write_tfrecord(filename: str, 
                   num_samples: int, 
                   dataset_config,
                   running_stats: RunningStatsDatapoints) -> None:
    """Writes the dataset into TFRecord file."""
    
    with tf.io.TFRecordWriter(filename) as writer:
        for _ in range(num_samples):
            # Generate data point
            parameters = generate_parameters(dataset_config['parameters'])
            
            feature, label = generate_datapoint(parameters)

            # Push the current samples for statistics calculation
            running_stats.update(feature, label)


            # print_data_info(feature, "feature")
            # feature type: <class 'numpy.ndarray'>
            # feature shape: (501, 3)
            # feature size: 1503
            # feature dtype: float64

            # print_data_info(label, "label")
            # label type: <class 'numpy.ndarray'>
            # label shape: (12,)
            # label size: 12
            # label dtype: float64

            # if normalization:
            #     # Normalization
            #     feature = normalize_data(feature, 
            #                             mean=dataset_config["statistics"]["most_accurate_mean_features"],
            #                             std=dataset_config["statistics"]["most_accurate_std_features"])
            #     label = normalize_data(label, 
            #                             mean=dataset_config["statistics"]["most_accurate_mean_labels"],
            #                             std=dataset_config["statistics"]["most_accurate_std_labels"])
                
            # if quantization:

            #     feature = quantize_data(feature)
            #     label = quantize_data(label)
            


            example = serialize_example(feature, label)
            writer.write(example)




def load_dataset(filename: Path, batch_size=32) -> tf.data.Dataset:
    """
    Load and parse the dataset from TFRecord files without shuffling.
    The data is returned in the same order as it appears in the TFRecord file.

    Args:
        filename (Path): The file path to the TFRecord file.

    Returns:
        tf.data.Dataset: A `tf.data.Dataset` object representing the input data.
    """

    raw_dataset = tf.data.TFRecordDataset(str(filename))

    # Apply the parsing function to each item in the raw dataset using the map method.
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Batch the data for easier consumption
    parsed_dataset = parsed_dataset.batch(batch_size)

    # Use prefetch to improve performance by overlapping the processing of data with data consumption.
    parsed_dataset = parsed_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Count the number of elements and batches
    num_elements, num_batches = count_elements_and_batches(parsed_dataset)

    print(f"Loaded dataset with {num_elements} elements across {num_batches} batches from {filename}")

    return parsed_dataset

def count_elements_and_batches(dataset: tf.data.Dataset) -> tuple:
    """
    Count the number of elements and batches in a TensorFlow dataset.

    Args:
        dataset (tf.data.Dataset): The dataset to count.

    Returns:
        tuple: A tuple containing the total number of elements and the total number of batches.
    """
    element_count = 0
    batch_count = 0

    for batch in dataset:
        # Count each batch
        batch_count += 1

        # Since the dataset is batched, we need to count the number of items in each batch.
        element_count += tf.shape(batch[0])[0]  # Assumes each batch is a tuple of (data, labels)

    return element_count, batch_count


def create_dataset(num_samples: int, 
                   dataset_config: dict, 
                   output_file_path: Path,
                   running_stats: RunningStatsDatapoints) -> Path:
    """
    Create a dataset suitable for feeding into a neural network.

    Parameters:
    num_samples (int): Number of samples to generate for the dataset.
    dataset_config (dict): Configuration parameters for dataset generation.
    output_file_path (Path): Full file path where the dataset will be stored. The directory will be created if it does not exist.

    Returns:
    Path: The path where the dataset file is stored.
    """

    # Set the seed for reproducibility
    seed_value = dataset_config.get('dataset', {}).get('seed', None)
    print(f"Setting seed {seed_value}")

    if seed_value is not None:
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

    # Extract directory from the file path and create it if it doesn't exist
    output_dir = output_file_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    # Write data to TFRecord
    write_tfrecord(str(output_file_path), num_samples, dataset_config, running_stats)

    print(f"Data written to {output_file_path}")

    return output_file_path


