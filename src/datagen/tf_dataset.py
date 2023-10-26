import numpy as np
import tensorflow as tf
from pathlib import Path
import math
from src.datagen.michaelis import generate_datapoint, generate_parameters

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



def transform_example(example: tf.Tensor, normalize, quantize, feature_stats, label_stats, num_quantization_bins) -> tf.train.Example:
        feature, label = parse_tfrecord_fn(example)

        if normalize:
            feature = (feature - feature_stats["mean"]) / feature_stats["std_dev"]
            label = (label - label_stats["mean"]) / label_stats["std_dev"]

        if quantize:
            feature_min_range = feature_stats["min"]
            feature_max_range = feature_stats["max"]
            label_min_range = label_stats["min"]
            label_max_range = label_stats["max"]
            # Quantization step:
            # Calculate the quantization step size for features and labels
            feature_quantization_step = (feature_max_range - feature_min_range) / (
                num_quantization_bins - 1
            )
            label_quantization_step = (label_max_range - label_min_range) / (
                num_quantization_bins - 1
            )

            # Quantize features and labels by mapping the original values to the corresponding quantization bins
            feature = tf.math.round(
                (feature - feature_min_range) / feature_quantization_step
            )
            label = tf.math.round((label - label_min_range) / label_quantization_step)

            # Clip the values to ensure they are within the range [0, num_quantization_bins - 1]
            feature = tf.clip_by_value(feature, 0, num_quantization_bins - 1)
            label = tf.clip_by_value(label, 0, num_quantization_bins - 1)

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
    feature_stats: dict,
    label_stats: dict,
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
            serialized_example = transform_example(raw_record, normalize, quantize, feature_stats, label_stats, num_quantization_bins)
            writer.write(serialized_example)

    print(f"Transformed dataset saved to {output_dataset_path}")



# ----------------------
# Main Dataset Sharded Functions
# ----------------------



def create_sharded_tfrecord(base_file_path: Path, data_config, running_stats):
    """
    Writes the dataset into multiple TFRecord files (shards).
    """
    num_shards = data_config["dataset"]["n_shards"]
    num_samples = data_config["dataset"]["n_samples"]
    samples_per_shard = math.ceil(num_samples / num_shards)

    for shard_id in range(num_shards):
        shard_file_path = base_file_path / f"shard_{shard_id}.tfrecord"
        with tf.io.TFRecordWriter(str(shard_file_path)) as writer:
            for _ in range(samples_per_shard):
                parameters = generate_parameters(data_config["parameters"])
                feature, label = generate_datapoint(parameters)
                running_stats.update(feature, label)
                tf_example = serialize_example(feature, label)
                writer.write(tf_example)
                
        print(f"Shard {shard_id} written at {shard_file_path}")

def load_sharded_tfrecord_dataset(base_file_path: Path, dataset_config: dict):
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
    feature_stats: dict,
    label_stats: dict,
    output_shards_directory: Path,
    shard_prefix: str,
    num_shards: int,
    normalize: bool = True,
    quantize: bool = False,
    num_quantization_bins: int = 256,
):
    """
    Applies transformations (normalization and/or quantization) to a sharded dataset and saves it as new sharded TFRecords.
    """

    # Ensure the output directory exists
    output_shards_directory.mkdir(parents=True, exist_ok=True)

    # Process each shard
    for shard_id in range(num_shards):
        # Construct the file paths for the original and new shards
        original_shard_path = input_shards_directory / f"{shard_prefix}_{shard_id}.tfrecord"
        output_shard_path = output_shards_directory / f"{shard_prefix}_{shard_id}.tfrecord"

        with tf.io.TFRecordWriter(str(output_shard_path)) as writer:
            raw_dataset = tf.data.TFRecordDataset(str(original_shard_path))
            for raw_record in raw_dataset:
                serialized_example = transform_example(raw_record, normalize, quantize, feature_stats, label_stats, num_quantization_bins)
                writer.write(serialized_example)

        print(f"Transformed shard {shard_id} saved to {output_shard_path}")

    print(f"All shards have been transformed and saved to {output_shards_directory}")
