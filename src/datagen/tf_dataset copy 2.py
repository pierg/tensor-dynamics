import numpy as np
import tensorflow as tf
from pathlib import Path
import math
from src.datagen.michaelis import generate_datapoint, generate_parameters
from src.datagen.runningstats import RunningStatsDatapoints
from src.shared.utils import save_dict_to_json_file
import pickle
from typing import Dict, Any


# ----------------------
# Utility Functions
# ----------------------


def _bytes_feature(value: np.ndarray) -> tf.train.Feature:
    """Returns a bytes_list from a tensor."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def serialize_example(feature: np.ndarray, label: np.ndarray) -> bytes:
    """
    Serializes a single (feature, label) pair into a tf.train.Example message.
    """
    feature_set = {
        "feature_shape": _bytes_feature(np.array(feature.shape, dtype=np.int64)),
        "feature_data": _bytes_feature(feature.astype(np.float64)),
        "label_shape": _bytes_feature(np.array(label.shape, dtype=np.int64)),
        "label_data": _bytes_feature(label.astype(np.float64)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_set))
    return example_proto.SerializeToString()


def parse_tfrecord_fn(example: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Parse TFRecord format.
    """
    feature_description = {
        "feature_shape": tf.io.FixedLenFeature([], tf.string),
        "feature_data": tf.io.FixedLenFeature([], tf.string),
        "label_shape": tf.io.FixedLenFeature([], tf.string),
        "label_data": tf.io.FixedLenFeature([], tf.string),
    }

    # Parse and reshape
    example = tf.io.parse_single_example(example, feature_description)
    feature_shape = tf.io.parse_tensor(example["feature_shape"], out_type=tf.int64)
    feature = tf.io.parse_tensor(example["feature_data"], out_type=tf.float64)
    feature = tf.reshape(feature, feature_shape)
    label_shape = tf.io.parse_tensor(example["label_shape"], out_type=tf.int64)
    label = tf.io.parse_tensor(example["label_data"], out_type=tf.float64)
    label = tf.reshape(label, label_shape)

    return feature, label


def create_datasets(base_directory: Path,
                    total_samples: int,
                    total_shards: int,
                    split_ratios: Tuple[float, float, float],  # Explicitly expecting three values
                    gen_parameters: dict,
                    apply_transformations: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create datasets and potentially apply transformations.

    :param base_directory: Base directory for the datasets.
    :param total_samples: Total number of samples.
    :param total_shards: Total number of shards.
    :param split_ratios: Ratios for splitting the datasets (training, testing, validation).
    :param gen_parameters: Parameters for data generation.
    :param apply_transformations: Flag to apply data transformations.
    :return: Statistics for original and transformed datasets.
    """
    # Validate the inputs
    if sum(split_ratios) != 1.0 or len(split_ratios) != 3:
        raise ValueError("There must be exactly three split ratios for training, testing, and validation datasets, and they must sum to 1.0.")

    # Prepare directories
    base_directory.mkdir(parents=True, exist_ok=True)
    base_origin_path = base_directory / "origin"
    base_origin_path.mkdir(exist_ok=True)

    base_transformed_path = None
    if apply_transformations:
        base_transformed_path = base_directory / "transformed"
        base_transformed_path.mkdir(exist_ok=True)

    dataset_categories = ['training', 'testing', 'validation']

    # Calculate the number of samples and shards for each split
    splits_info = {
        category: {
            'n_samples': int(total_samples * ratio),
            'n_shards': max(1, int(total_shards * ratio))  # At least 1 shard
        }
        for category, ratio in zip(dataset_categories, split_ratios)
    }

    print("Creating datasets...")

    running_stats_original = {}
    running_stats_transformed = {} if apply_transformations else None

    # Generate datasets
    for dataset_type, info in splits_info.items():
        # Original dataset
        origin_directory = base_origin_path / dataset_type
        origin_directory.mkdir(parents=True, exist_ok=True)

        print(f"Creating {dataset_type} dataset with {info['n_samples']} samples across {info['n_shards']} shards...")
        running_stats_original[dataset_type] = create_sharded_tfrecord(
            base_file_path=origin_directory,
            gen_parameters=gen_parameters,
            num_shards=info['n_shards'],
            num_samples=info['n_samples']
        )

        # Transformed dataset
        if apply_transformations:
            transformed_directory = base_transformed_path / dataset_type
            transformed_directory.mkdir(parents=True, exist_ok=True)

            running_stats_transformed[dataset_type] = transform_and_save_sharded_dataset(
                input_shards_directory=origin_directory,
                output_shards_directory=transformed_directory,
                original_running_stats=running_stats_original[dataset_type]
            )

    print("All datasets have been created.")
    return running_stats_original, running_stats_transformed



def create_sharded_tfrecord(base_file_path: Path, 
                            gen_parameters: dict, 
                            num_shards: int, 
                            num_samples: int):
    running_stats = RunningStatsDatapoints()  # Assuming this class is properly defined elsewhere
    samples_per_shard = math.ceil(num_samples / num_shards)

    for shard_id in range(num_shards):
        shard_file_path = base_file_path / f"shard_{shard_id}.tfrecord"
        with tf.io.TFRecordWriter(str(shard_file_path)) as writer:
            for _ in range(samples_per_shard):
                parameters = generate_parameters(gen_parameters)
                feature, label = generate_datapoint(parameters)
                running_stats.update(feature, label)
                tf_example = serialize_example(feature, label)
                writer.write(tf_example)

        print(f"Shard {shard_id} written at {shard_file_path}")

    save_dict_to_json_file(running_stats.to_dict(), base_file_path / "stats.json", )

    # Save running stats as a pickle
    with open(base_file_path / "stats.pkl", "wb") as pkl_file:
        pickle.dump(running_stats, pkl_file)

    return running_stats


def load_datasets(base_directory: Path, 
                  batch_size: int = 32):
    datasets = {}
    running_stats = {}

    for dataset_type in ['training', 'testing', 'validation']:
        dataset_directory = base_directory / dataset_type
        print(f"Loading {dataset_type} dataset...")
        
        # Load the dataset
        datasets[dataset_type] = load_all_shards_tfrecord_dataset(dataset_directory, batch_size)

        # Load running stats
        stats_path = dataset_directory / "stats.pkl"
        with open(stats_path, "rb") as pkl_file:
            running_stats[dataset_type] = pickle.load(pkl_file)

    return datasets, running_stats



def load_all_shards_tfrecord_dataset(base_file_path: Path, 
                                     batch_size: int = 32) -> tf.data.Dataset:
    """
    Create a dataset by loading all shards without splitting and without loading the full dataset into memory.
    """

    # Ensure the base_file_path exists and is a directory.
    if not base_file_path.is_dir():
        raise ValueError(f"{base_file_path} is not a valid directory.")

    # Collect all .tfrecord files within the directory.
    tfrecord_files = list(base_file_path.glob('*.tfrecord'))

    # If there are no .tfrecord files, raise an error.
    if not tfrecord_files:
        raise FileNotFoundError(f"No .tfrecord files found in {base_file_path}.")

    num_shards = len(tfrecord_files)

    # Generate file paths for all the shards.
    all_files = [str(file_path) for file_path in tfrecord_files]

    # Utilize the 'interleave' function to read from multiple files in parallel, effectively shuffling the dataset records.
    full_dataset = tf.data.TFRecordDataset(all_files).interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=num_shards,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Parse, batch, and prefetch the datasets.
    full_dataset = full_dataset.map(
        parse_tfrecord_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return full_dataset

