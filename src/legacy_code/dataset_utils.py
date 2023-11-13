from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset.statistics_generator import RunningStatsDatapoints
from dataset.tf_data_uilities import load_tfrecord_dataset


def print_data_info(data: np.ndarray, label: str) -> None:
    """Display formatted information about the numpy data.

    Args:
        data (np.ndarray): Data to display information about.
        label (str): Descriptor for the data (e.g., "Data" or "Predictions").
    """
    info_strings = [
        f"\n{label} Info:",
        f"Type : {type(data)}",
        f"Shape: {data.shape}",
        f"Size : {data.size}",
        f"Dtype: {data.dtype}",
        "-" * 50,
    ]
    print("\n".join(info_strings))


def compute_and_print_dataset_statistics(dataset: tf.data.Dataset, name: str) -> None:
    """Compute and print statistics for a tf.data.Dataset, and use print_data_info for additional information.

    Args:
        dataset (tf.data.Dataset): The input dataset to analyze.
    """
    # First, we need to convert our dataset's data into a NumPy array. We'll assume that our dataset is batched.
    # If it's not batched, we should batch it ourselves or handle individual items.

    all_features = []
    all_labels = []
    for features, labels in dataset:
        # Convert tensors to numpy arrays and append to our lists
        all_features.append(features.numpy())
        all_labels.append(labels.numpy())

    # Concatenate all the data together
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # You can now compute statistics like mean, std, etc., for your features and labels here.
    features_mean = np.mean(all_features, axis=0)
    features_std = np.std(all_features, axis=0)
    labels_mean = np.mean(all_labels, axis=0)
    labels_std = np.std(all_labels, axis=0)
    print("\n\nDataset: {name}")

    # Now, let's print the info of the features and labels
    print_data_info(all_features, label="Features")
    print_data_info(all_labels, label="Labels")

    # Print computed statistics
    print("\nComputed Statistics:")
    print(f"Features Mean: {features_mean}")
    print(f"Features Std : {features_std}")
    print(f"Labels Mean  : {labels_mean}")
    print(f"Labels Std   : {labels_std}")
    print("-" * 50)


def compute_statistics(dataset: tf.data.Dataset) -> tuple:
    """
    Compute statistical metrics (mean, variance, and range) of target values in a dataset.

    Args:
        dataset (tf.data.Dataset): Dataset containing data points and target values.

    Returns:
        tuple: Statistical metrics - mean, variance, and range of target values.
    """
    targets = np.concatenate([target.numpy() for _, target in dataset], axis=0)

    # Calculate statistics
    mean_abs = np.mean(np.absolute(targets))
    variance = np.var(targets)
    range_value = np.ptp(targets)  # ptp() returns the range (max - min) of values.

    return mean_abs, variance, range_value


def pretty_print_sharded_dataset_elements(
    shards_directory: Path,
    shard_prefix: str,
    num_shards: int,
    running_stats: RunningStatsDatapoints = None,
    compute_stats: bool = True,
    print_elements: bool = False,
    n: int = None,
) -> None:
    """
    Pretty prints 'n' elements from a sharded dataset located at the provided directory.
    It utilizes running statistics if provided.

    Args:
        shards_directory (Path): The directory where the dataset shards are located.
        shard_prefix (str): The common prefix of the shard files.
        num_shards (int): The number of shards in the dataset.
        running_stats (RunningStatsDatapoints, optional): Running statistics for features and labels.
        n (int, optional): The number of records to print from the dataset. Prints all if None.
    """

    # If running_stats or compute_stats is enabled, initialize variables for statistics calculation
    if running_stats or compute_stats:
        total_feature_stats = []
        total_label_stats = []

    if print_elements:
        # Iterate over each shard
        for shard_id in range(num_shards):
            shard_path = shards_directory / f"{shard_prefix}_{shard_id}.tfrecord"
            dataset = load_tfrecord_dataset(shard_path, batch_size=1)

            # Perform operations specific to running_stats or compute_stats if they are enabled
            if running_stats or compute_stats:
                # Accumulate statistics from this shard
                for features, labels in dataset:
                    if running_stats:
                        running_stats.update(features, labels)

                    if compute_stats:
                        # Accumulate features and labels for later statistics computation
                        total_feature_stats.append(features.numpy())
                        total_label_stats.append(labels.numpy())

            # Pretty print 'n' elements from the current shard
            num_elements = n if n is not None else dataset.cardinality().numpy()
            for count, (features, labels) in enumerate(
                dataset.take(num_elements), start=1
            ):
                print(f"\nShard: {shard_id}, Element: {count}")
                print("Feature:", features.numpy(), sep="\n")
                print("Label:", labels.numpy(), sep="\n")
                print("-" * 30)

    # If running_stats was used, print the accumulated statistics
    if running_stats:
        feature_stats = running_stats.get_feature_stats()
        label_stats = running_stats.get_label_stats()

        print("\nAccumulated Feature Statistics:")
        print(f"Mean     : {feature_stats['mean']}")
        print(f"Variance : {feature_stats['variance']}")
        print(f"Std Dev  : {feature_stats['std_dev']}")
        print("-" * 30)

        print("\nAccumulated Label Statistics:")
        print(f"Mean     : {label_stats['mean']}")
        print(f"Variance : {label_stats['variance']}")
        print(f"Std Dev  : {label_stats['std_dev']}")
        print("-" * 30)

    # If compute_stats was used, calculate and print the statistics from the accumulated features and labels
    if compute_stats and total_feature_stats and total_label_stats:
        # Calculate statistics here from total_feature_stats and total_label_stats
        # For simplicity, this example does not actually compute the statistics
        # Replace the following print statement with actual statistics computation and printing
        print(
            "\nComputed statistics for the entire sharded dataset would be displayed here."
        )


def pretty_print_dataset_elements(
    dataset_location: Path,
    running_stats: RunningStatsDatapoints = None,
    compute_stats: bool = True,
    n: int = None,
) -> None:
    """
    Pretty prints 'n' elements from the dataset located at the provided path.
    It utilizes running statistics if provided.

    Args:
        dataset_location (Path): The file path of the TFRecord dataset.
        running_stats (RunningStatsDatapoints, optional): Running statistics for features and labels.
        n (int, optional): The number of records to print from the dataset. Prints all if None.
    """
    dataset = load_tfrecord_dataset(dataset_location, batch_size=1)

    # If running_stats is provided, print the statistics from it
    if running_stats is not None:
        feature_stats = running_stats.get_feature_stats()
        label_stats = running_stats.get_label_stats()

        print("\nFeature Statistics:")
        print(f"Mean     : {feature_stats['mean']}")
        print(f"Variance : {feature_stats['variance']}")
        print(f"Std Dev  : {feature_stats['std_dev']}")
        print("-" * 30)

        print("\nLabel Statistics:")
        print(f"Mean     : {label_stats['mean']}")
        print(f"Variance : {label_stats['variance']}")
        print(f"Std Dev  : {label_stats['std_dev']}")
        print("-" * 30)

    if compute_stats:
        # If running_stats is not provided, calculate and print statistics from the dataset
        mean, variance, range_value = compute_statistics(dataset)
        print(
            f"\nComputing Dataset Statistics:\nRange: {range_value}\nMean : {mean}\nVar  : {variance}\n"
            + "-" * 30
        )

    # Use the cardinality of the dataset if 'n' is not specified
    num_elements = n if n is not None else dataset.cardinality().numpy()

    for count, (features, labels) in enumerate(dataset.take(num_elements), start=1):
        print(f"\nElement: {count}")
        print("Feature:", features.numpy(), sep="\n")
        print("Label:", labels.numpy(), sep="\n")
        print("-" * 30)


def count_elements_and_batches(dataset: tf.data.Dataset) -> tuple:
    """Counts the number of elements and batches in a TensorFlow dataset.

    Args:
        dataset (tf.data.Dataset): The dataset to count.

    Returns:
        tuple: Total number of elements and total number of batches.
    """
    num_elements = dataset.reduce(
        0, lambda x, _: x + 1
    ).numpy()  # More efficient counting

    # Assumes the dataset might already be batched
    num_batches = tf.data.experimental.cardinality(dataset).numpy()
    num_batches = (
        num_batches if num_batches != -1 else sum(1 for _ in dataset)
    )  # Handle unknown cardinality

    return num_elements, num_batches


def get_dataset_statistics(dataset, dataset_name):
    """
    Compute and print basic statistics about a TensorFlow dataset.
    This function assumes that the dataset is batched.

    Args:
    dataset (tf.data.Dataset): The dataset to compute statistics for.
    dataset_name (str): A human-readable name for the dataset to print.
    """
    # We will go through only one batch to get the shapes and types of the dataset features.
    # Note: We're assuming here that all samples have the same shape and the dataset is batched.
    for features in dataset.take(1):
        if isinstance(features, (tuple, list)) and len(features) == 2:
            data, labels = features
        else:
            data = features  # No labels in the dataset, only features.

        print(f"\nStatistics for {dataset_name}: ")

        # If the data is a dictionary of features (which is common with TFRecords), we iterate through each feature.
        if isinstance(data, dict):
            for feature_name, feature_tensor in data.items():
                print(f"Feature: {feature_name}")
                print(f"    Type: {feature_tensor.dtype}")
                print(
                    f"    Shape: {feature_tensor.shape}"
                )  # The first dimension is the batch size.
        else:  # If the data is a single tensor, just print its type and shape.
            print(f"Data Type: {data.dtype}")
            print(f"Data Shape: {data.shape}")  # The first dimension is the batch size.

        # If labels are present, print their type and shape.
        if "labels" in locals():
            print(f"Label Type: {labels.dtype}")
            print(
                f"Label Shape: {labels.shape}"
            )  # The first dimension is the batch size.
