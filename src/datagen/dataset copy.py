import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import toml
from typing import Tuple, List
from src.datagen.michaelis import generate_datapoint, generate_parameters

def _bytes_feature(value: np.ndarray) -> tf.train.Feature:
    """Returns a bytes_list from a numpy array / byte."""
    # Convert EagerTensor to byte string before placing inside BytesList
    value = tf.io.serialize_tensor(value).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(feature: np.ndarray, label: np.ndarray) -> bytes:
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature_set = {
        'feature': _bytes_feature(feature),
        'label': _bytes_feature(label),
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_set))
    return example_proto.SerializeToString()

def write_tfrecord(filename: str, num_samples: int, dataset_config) -> None:
    """Writes the dataset into TFRecord file."""
    
    with tf.io.TFRecordWriter(filename) as writer:
        for _ in range(num_samples):
            # Generate data point
            parameters = generate_parameters(dataset_config['parameters'])
            feature, label = generate_datapoint(parameters)

            example = serialize_example(feature, label)
            writer.write(example)


def parse_tfrecord_fn(example: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parse TFRecord format
    """
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    feature = tf.io.parse_tensor(example['feature'], out_type=tf.float32)
    label = tf.io.parse_tensor(example['label'], out_type=tf.float32)
    
    return feature, label

def load_dataset(filename: str, buffer_size: int) -> tf.data.Dataset:
    """
    Load and parse the dataset from TFRecord file
    """
    raw_dataset = tf.data.TFRecordDataset(filename)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    parsed_dataset = parsed_dataset.shuffle(buffer_size).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return parsed_dataset


def create_dataset(num_samples: int, dataset_config: dict, output_dir: Path) -> tf.data.Dataset:
    """
    Create a dataset suitable for feeding into a neural network.

    Parameters:
    num_samples (int): Number of samples to generate for the dataset.
    dataset_config (dict): Configuration parameters for dataset generation.
    output_dir (str): Directory where the dataset will be stored.

    Returns:
    tf.data.Dataset: A TensorFlow Dataset object ready for training.
    """

    # Set the seed for reproducibility
    seed_value = dataset_config.get('seed', {}).get('value', None)
    print(f"Setting seed {seed_value}")

    if seed_value is not None:
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

    # Set up the TFRecord file path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    tfrecord_file = output_dir / 'dataset.tfrecord'

    # Write data to TFRecord
    write_tfrecord(str(tfrecord_file), num_samples, dataset_config)

    print(f"Data written to {tfrecord_file}")


if __name__ == "__main__":
    # Set paths and load configurations
    current_dir = Path(__file__).resolve().parent
    data_config_file = current_dir / "data.toml"
    output_folder = current_dir / "output"

    data_config = toml.load(data_config_file)
    dataset_config = data_config["d1"]  # Assuming "d1" is a placeholder for your actual config key

    # Generate, save
    create_dataset(1000, dataset_config, output_folder)  # Set the desired number of samples

