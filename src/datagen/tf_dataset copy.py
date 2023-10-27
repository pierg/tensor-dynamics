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



def transform_example(example: tf.Tensor, 
                      normalize: bool, 
                      quantize: bool, 
                      original_running_stats: RunningStatsDatapoints, 
                      transformed_running_stats: RunningStatsDatapoints, 
                      num_quantization_bins: int) -> tf.train.Example:
    # Parse the example to get the feature and label
    feature, label = parse_tfrecord_fn(example)

    original_feature_stats = original_running_stats.get_feature_stats()
    original_label_stats = original_running_stats.get_label_stats()

    if normalize:
        # Normalizing the features and labels
        feature = (feature - original_feature_stats["full"]["mean"]) / original_feature_stats["full"]["std_dev"]
        label = (label - original_label_stats["full"]["mean"]) / original_label_stats["full"]["std_dev"]

    if quantize:
        # Setting the range for quantization
        feature_min_range = original_feature_stats["full"]["min"]
        feature_max_range = original_feature_stats["full"]["max"]
        label_min_range = original_label_stats["full"]["min"]
        label_max_range = original_label_stats["full"]["max"]

        # Calculate the quantization step size for features and labels
        feature_quantization_step = (feature_max_range - feature_min_range) / (num_quantization_bins - 1)
        label_quantization_step = (label_max_range - label_min_range) / (num_quantization_bins - 1)

        # Quantize features and labels by mapping the original values to the corresponding quantization bins
        feature = tf.math.round((feature - feature_min_range) / feature_quantization_step)
        label = tf.math.round((label - label_min_range) / label_quantization_step)

        # Clip the values to ensure they are within the valid range
        feature = tf.clip_by_value(feature, 0, num_quantization_bins - 1)
        label = tf.clip_by_value(label, 0, num_quantization_bins - 1)

    # Update the running statistics for the transformed data
    transformed_running_stats.update(feature.numpy(), label.numpy())

    # Serialize the example for writing to TFRecord
    return serialize_example(feature.numpy(), label.numpy())



# ----------------------
# Main Dataset Functions
# ----------------------


def write_tfrecord(
    file_path: Path, num_samples: int, dataset_config, running_stats
) -> None:
    """
    Writes dataset into a TFRecord file.
    """
    with tf.io.TFRecordWriter(str(file_path)) as writer:
        for _ in range(num_samples):
            parameters = generate_parameters(dataset_config["parameters"])
            feature, label = generate_datapoint(parameters)
            running_stats.update(feature, label)
            tf_example = serialize_example(feature, label)
            writer.write(tf_example)
    print(f"Dataset generation and writing to TFRecord completed: {file_path}")


def load_tfrecord_dataset(file_path: Path, batch_size: int) -> tf.data.Dataset:
    """
    Reads a TFRecord file and returns a batched dataset.
    """
    raw_dataset = tf.data.TFRecordDataset([str(file_path)])
    parsed_dataset = raw_dataset.map(
        parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    parsed_dataset = parsed_dataset.batch(batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    return parsed_dataset


def transform_and_save_dataset(
    original_dataset_path: Path,
    original_running_stats: RunningStatsDatapoints,
    transformed_running_stats: RunningStatsDatapoints,
    output_dataset_path: Path,
    normalize: bool = True,
    quantize: bool = False,
    num_quantization_bins: int = 256,
):
    """
    Applies transformations (normalization and/or quantization) to the dataset and saves it as a new TFRecord.
    """


    with tf.io.TFRecordWriter(str(output_dataset_path)) as writer:
        raw_dataset = tf.data.TFRecordDataset(str(original_dataset_path))
        for raw_record in raw_dataset:
            serialized_example = transform_example(raw_record, normalize, quantize, original_running_stats, transformed_running_stats, num_quantization_bins)
            writer.write(serialized_example)

    print(f"Transformed dataset saved to {output_dataset_path}")



# ----------------------
# Main Dataset Sharded Functions
# ----------------------

from pathlib import Path
from typing import Dict, Tuple


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


def create_sharded_tfrecord_old(base_file_path: Path, 
                            gen_parameters: dict,
                            num_shards, 
                            num_samples):
    """
    Writes the dataset into multiple TFRecord files (shards).
    """
    running_stats = RunningStatsDatapoints()
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
    # picke running_stats and save it in stats.pk

    return running_stats
    

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

def load_sharded_tfrecord_dataset_and_split(base_file_path: Path, dataset_config: dict):
    """
    Create training, validation, and test datasets by selecting shards without loading the full dataset into memory.
    """
    num_shards = dataset_config["n_shards"]
    split_ratios = dataset_config["splits"]
    batch_size = dataset_config["batch_size"]

    # Ensure there are enough shards.
    if num_shards < 3:  # minimum for train, val, and test
        raise ValueError("Not enough shards to create train/val/test split.")

    # Shuffle the shards to randomize the dataset parts used for different splits (if required).
    shard_ids = tf.random.shuffle(tf.range(num_shards))

    # Calculate how many shards will be used for each dataset split, ensuring at least one shard per split.
    train_shards = max(1, int(num_shards * split_ratios[0]))
    val_shards = max(1, int(num_shards * split_ratios[1]))
    # Remaining shards for testing (ensuring we use all shards).
    test_shards = num_shards - train_shards - val_shards

    # Guard against having zero shards for testing
    if test_shards <= 0:
        raise ValueError("Not enough shards to allocate for testing.")

    # Generate file paths for the shards.
    train_files = [str(base_file_path / f"shard_{shard_id}.tfrecord") for shard_id in shard_ids[:train_shards]]
    val_files = [str(base_file_path / f"shard_{shard_id}.tfrecord") for shard_id in shard_ids[train_shards:train_shards+val_shards]]
    test_files = [str(base_file_path / f"shard_{shard_id}.tfrecord") for shard_id in shard_ids[train_shards+val_shards:]]

    # Utilize the 'interleave' function to read from multiple files in parallel, effectively shuffling the dataset records.
    train_dataset = tf.data.TFRecordDataset(train_files).interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=train_shards, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = tf.data.TFRecordDataset(val_files).interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=val_shards, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = tf.data.TFRecordDataset(test_files).interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=test_shards, num_parallel_calls=tf.data.AUTOTUNE)

    # Parse, batch, and prefetch the datasets.
    train_dataset = train_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = test_dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset




def transform_and_save_sharded_dataset(
    input_shards_directory: Path,
    output_shards_directory: Path,
    original_running_stats: RunningStatsDatapoints,
    normalize: bool = True,
    quantize: bool = False,
    num_quantization_bins: int = 256,
) -> RunningStatsDatapoints:
    """
    Applies transformations (normalization and/or quantization) to a sharded dataset 
    and saves it as new sharded TFRecords. Automatically detects and processes all
    .tfrecord files in the input directory.
    """

    # Ensure the output directory exists
    output_shards_directory.mkdir(parents=True, exist_ok=True)

    transformed_running_stats = RunningStatsDatapoints()  # Assuming this initializes empty stats

    # Find all .tfrecord files in the input directory
    input_tfrecord_files = input_shards_directory.glob('*.tfrecord')

    # Process each shard file
    for input_file in input_tfrecord_files:
        if input_file.is_file():
            # Define the output file path. This assumes the desire to keep the original file name.
            output_file = output_shards_directory / input_file.name

            with tf.io.TFRecordWriter(str(output_file)) as writer:
                raw_dataset = tf.data.TFRecordDataset(str(input_file))
                for raw_record in raw_dataset:
                    serialized_example = transform_example(
                        raw_record, 
                        normalize, 
                        quantize, 
                        original_running_stats, 
                        transformed_running_stats, 
                        num_quantization_bins
                    )
                    writer.write(serialized_example)

            print(f"Transformed shard {input_file.name} saved to {output_file}")

    print(f"All shards have been transformed and saved to {output_shards_directory}")

    save_dict_to_json_file(transformed_running_stats.to_dict(), output_shards_directory / "stats.json", )

    # Save running stats as a pickle
    with open(output_shards_directory / "stats.pkl", "wb") as pkl_file:
        pickle.dump(transformed_running_stats, pkl_file)

    return transformed_running_stats