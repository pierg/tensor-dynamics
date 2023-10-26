import tensorflow as tf
import numpy as np
from pathlib import Path
from src.datagen.runningstats import RunningStatsDatapoints
from src.datagen.tf_dataset import write_tfrecord



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


