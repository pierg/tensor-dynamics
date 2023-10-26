


from pathlib import Path
from src.datagen.dataset_utils import pretty_print_sharded_dataset_elements
from src.datagen.runningstats import RunningStatsDatapoints
from src.datagen.tf_dataset import create_sharded_tfrecord, transform_and_save_sharded_dataset
from typing import Any, Dict

def create_dataset(base_file_path: Path, data_config: Dict[str, Any]) -> Path:
    # Ensure the base directory exists.
    base_file_path.mkdir(parents=True, exist_ok=True)

    # Prepare directories for the original and normalized datasets.
    base_origin_path = base_file_path / "origin"
    base_norm_path = base_file_path / "normalized"
    base_origin_path.mkdir(exist_ok=True)
    base_norm_path.mkdir(exist_ok=True)

    # Constants for the sharding process.
    shard_prefix = "shard"
    num_shards = data_config.get('dataset', {}).get('n_shards')
    if num_shards is None:
        raise ValueError("Number of shards not defined in configuration")

    # Initialize running statistics for the datapoints.
    running_stats = RunningStatsDatapoints()

    # Create the original dataset in shards and collect statistics.
    create_sharded_tfrecord(base_origin_path, data_config, running_stats)

    # Display examples from the original dataset.
    print("\nOriginal Dataset:")
    pretty_print_sharded_dataset_elements(
        shards_directory=base_origin_path,
        shard_prefix=shard_prefix,
        num_shards=num_shards,
        running_stats=running_stats,
        n=5
    )

    # Normalize the dataset based on collected statistics and save in shards.
    feature_stats = running_stats.get_feature_stats()
    label_stats = running_stats.get_label_stats()
    transform_and_save_sharded_dataset(
        input_shards_directory=base_origin_path,
        feature_stats=feature_stats,
        label_stats=label_stats,
        output_shards_directory=base_norm_path,
        shard_prefix=shard_prefix,
        num_shards=num_shards,
    )

    # Display examples from the normalized dataset.
    print("\nNormalized Dataset:")
    pretty_print_sharded_dataset_elements(
        shards_directory=base_norm_path,
        shard_prefix=shard_prefix,
        num_shards=num_shards,
        running_stats=running_stats,
        n=5
    )

    # Return the path of the normalized dataset.
    return base_norm_path
