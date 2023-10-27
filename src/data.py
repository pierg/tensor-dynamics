import hashlib
import toml
from pathlib import Path
from typing import Dict, Any
import shutil
from src.datagen.tf_dataset import create_datasets, load_datasets
from src.shared.utils import pretty_print_dict, save_dict_to_json_file
from typing import Dict, Any
from pathlib import Path
import shutil


def generate_dataset_id(seed: int, n_samples: int, n_shards: int) -> str:
    hash_object = hashlib.sha256(f"{seed}{n_samples}{n_shards}".encode())
    return hash_object.hexdigest()[:5]


def load_all_datasets(
    dataset_config: Dict[str, Any], data_base_folder: Path, transformed: bool = True
) -> tuple[dict, dict]:
    batch_size = dataset_config["dataset"]["batch_size"]
    if transformed:
        return load_datasets(data_base_folder / "transformed", batch_size)
    return load_datasets(data_base_folder / "origin", batch_size)


def process_datasets(
    dataset_config: Dict[str, Any], datasets_folder: Path, overwrite: bool = False
) -> Path:
    """
    Process the datasets based on the provided configuration.

    :param dataset_config: Configuration details for the dataset.
    :param datasets_folder: Base directory for the datasets.
    :param overwrite: Whether to overwrite an existing dataset.
    :return: Path of the base folder where the dataset is stored.
    """
    c_dataset = dataset_config["dataset"]
    c_parameters = dataset_config["parameters"]

    # Generate unique dataset ID
    dataset_id = generate_dataset_id(
        c_dataset["seed"], c_dataset["n_samples"], c_dataset["n_shards"]
    )
    base_folder_path = datasets_folder / f"dataset_{dataset_id}"

    if base_folder_path.exists() and overwrite:
        print(f"Overwriting existing dataset with dataset_id '{dataset_id}'...")
        shutil.rmtree(base_folder_path, ignore_errors=True)

    # Prepare the base folder for the dataset
    base_folder_path.mkdir(parents=True, exist_ok=True)

    # Save the dataset configuration
    config_path = base_folder_path / "dataset_config.json"
    save_dict_to_json_file(dataset_config, config_path)

    # Create datasets and retrieve statistics
    try:
        running_stats_original, running_stats_transformed = create_datasets(
            base_folder_path,
            c_dataset["n_samples"],
            c_dataset["n_shards"],
            dataset_config.get(
                "splits", [0.7, 0.15, 0.15]
            ),  # default split ratios if not provided
            c_parameters,
            apply_transformations=True,
        )
    except Exception as e:
        print(f"Error during dataset creation: {e}")
        raise

    # Save all statistics in an info.json file within the dataset configuration

    info_to_print = {
        "original_stats": {
            key: stats.to_dict(reduced=True)
            for key, stats in running_stats_original.items()
        },
        "transformed_stats": {
            key: stats.to_dict(reduced=True)
            for key, stats in running_stats_transformed.items()
        },
    }

    pretty_print_dict(info_to_print)

    info_complete = {
        "original_stats": {
            key: stats.to_dict() for key, stats in running_stats_original.items()
        },
        "transformed_stats": {
            key: stats.to_dict() for key, stats in running_stats_transformed.items()
        },
    }

    info_path = base_folder_path / "datasets_info.json"
    save_dict_to_json_file(info_complete, info_path)

    print(f"\nAll datasets have been processed and saved in '{base_folder_path}'.")
    return base_folder_path


# The main execution can be wrapped in a main function and executed conditionally.
def main():
    from src.shared import (
        data_config_file,
        datasets_folder,
    )  # assuming the import path is correct

    dataset_config = toml.load(data_config_file)["d1"]
    process_datasets(dataset_config, datasets_folder, overwrite=False)


if __name__ == "__main__":
    main()
