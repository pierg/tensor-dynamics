import numpy as np
import tensorflow as tf
from pathlib import Path
from src.datagen.runningstats import RunningStatsDatapoints
from src.datagen.tf_dataset import load_tfrecord_dataset

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
        "-" * 50
    ]
    print('\n'.join(info_strings))

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



def pretty_print_dataset_elements(dataset_location: Path, 
                                  running_stats: RunningStatsDatapoints = None, 
                                  compute_stats: bool = True,
                                  n: int = None) -> None:
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
        print(f"\nComputing Dataset Statistics:\nRange: {range_value}\nMean : {mean}\nVar  : {variance}\n" + "-" * 30)

    # Use the cardinality of the dataset if 'n' is not specified
    num_elements = n if n is not None else dataset.cardinality().numpy()

    for count, (features, labels) in enumerate(dataset.take(num_elements), start=1):
        print(f"\nElement: {count}")
        print("Feature:", features.numpy(), sep='\n')
        print("Label:", labels.numpy(), sep='\n')
        print("-" * 30)

def count_elements_and_batches(dataset: tf.data.Dataset) -> tuple:
    """Counts the number of elements and batches in a TensorFlow dataset.

    Args:
        dataset (tf.data.Dataset): The dataset to count.

    Returns:
        tuple: Total number of elements and total number of batches.
    """
    num_elements = dataset.reduce(0, lambda x, _: x + 1).numpy()  # More efficient counting

    # Assumes the dataset might already be batched
    num_batches = tf.data.experimental.cardinality(dataset).numpy()
    num_batches = num_batches if num_batches != -1 else sum(1 for _ in dataset)  # Handle unknown cardinality

    return num_elements, num_batches
