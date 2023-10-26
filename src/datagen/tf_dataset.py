import numpy as np
import tensorflow as tf
from pathlib import Path

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

    def transform_example(example: tf.Tensor) -> tf.train.Example:
        feature, label = parse_tfrecord_fn(example)

        if normalize:
            feature = (feature - feature_stats["mean"]) / feature_stats["std_dev"]
            label = (label - label_stats["mean"]) / label_stats["std_dev"]

        if quantize:
            # print("Feature shape")
            # print(feature.shape)
            # print("Label shape")
            # print(label.shape)

            feature_min_range = feature_stats["min"]
            feature_max_range = feature_stats["max"]
            label_min_range = label_stats["min"]
            label_max_range = label_stats["max"]

            # print("Feature min max shape")
            # print(feature_min_range.shape)
            # print(feature_max_range.shape)

            # print("Labels min max shape")
            # print(label_min_range.shape)
            # print(label_max_range.shape)

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

    with tf.io.TFRecordWriter(str(output_dataset_path)) as writer:
        raw_dataset = tf.data.TFRecordDataset(str(original_dataset_path))
        for raw_record in raw_dataset:
            serialized_example = transform_example(raw_record)
            writer.write(serialized_example)

    print(f"Transformed dataset saved to {output_dataset_path}")
