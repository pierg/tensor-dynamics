'''
Author: Piergiuseppe Mallozzi
Date: November 2023
'''

from pathlib import Path

import toml
from generators.dataset_utils import pretty_print_dataset_elements

from dataset.statistics_generator import RunningStatsDatapoints
from dataset.tf_data_uilities import transform_and_save_dataset, write_tfrecord


def main() -> None:
    # Set paths and load configurations
    current_dir = Path(__file__).resolve().parent
    data_config_file = current_dir / "data.toml"

    data_config = toml.load(data_config_file)

    dataset_config = data_config["d1"]

    n_samples = dataset_config.get("dataset", {}).get("n_samples", None)

    output_dir = Path(__file__).parent.resolve() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    running_stats = RunningStatsDatapoints()

    # Create original dataset
    orig_dataset_path = output_dir / "orig_dataset.tfrecord"
    write_tfrecord(orig_dataset_path, n_samples, dataset_config, running_stats)
    print("\nOriginal Dataset:")
    pretty_print_dataset_elements(orig_dataset_path, running_stats)

    # Normalize the dataset
    norm_dataset_path = output_dir / "norm_dataset.tfrecord"
    transform_and_save_dataset(
        orig_dataset_path,
        running_stats.get_feature_stats(),
        running_stats.get_label_stats(),
        norm_dataset_path,
        normalize=True,
        quantize=False,
    )
    print("\nNormalized Dataset:")
    pretty_print_dataset_elements(norm_dataset_path, running_stats)

    # Quantize and normalize the dataset
    quant_norm_dataset_path = output_dir / "quant_norm_dataset.tfrecord"
    transform_and_save_dataset(
        orig_dataset_path,
        running_stats.get_feature_stats(),
        running_stats.get_label_stats(),
        quant_norm_dataset_path,
        normalize=True,
        quantize=True,
        num_quantization_bins=512,
    )
    print("\nQuantized and Normalized Dataset:")
    pretty_print_dataset_elements(quant_norm_dataset_path, running_stats)


if __name__ == "__main__":
    main()
