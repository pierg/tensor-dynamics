import numpy as np
import tensorflow as tf
from pathlib import Path
from src.datagen.dataset import *
import toml
import numpy as np
import tensorflow as tf
from src.datagen.runningstats import RunningStatsDatapoints


def main() -> None:

    # Set paths and load configurations
    current_dir = Path(__file__).resolve().parent
    data_config_file = current_dir / "data.toml"
    statistics_file = current_dir / "statistics.toml"
    
    data_config = toml.load(data_config_file)
    stats_config = toml.load(statistics_file)

    dataset_config = data_config["d1"] 
    stats_config = stats_config["d1"]["statistics"] 

    dataset_config["statistics"] = stats_config


    n_samples = dataset_config.get('dataset', {}).get('n_samples', None)


    output_dir = Path(__file__).parent.resolve() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize RunningStats instances for calculating statistics
    # We're initializing with zero-size dimensions as we don't know the exact shape until we generate data points.

    running_stats = RunningStatsDatapoints()

    dataset_path = output_dir / 'orig_dataset.tfrecord'

    create_dataset(n_samples, dataset_config, dataset_path, running_stats)
    print("\nOriginal")
    pretty_print_dataset_elements(dataset_path)

    feature_stats = running_stats.get_feature_stats()
    label_stats = running_stats.get_label_stats()


    transform_and_save_dataset(dataset_path, feature_stats, label_stats)
    
    # returns : {
    # #         "mean": self.labels.get_mean(),
    # #         "variance": self.labels.get_variance(),
    # #         "std_dev": self.labels.get_standard_deviation(),
    # #     }

    # # Create and save 'norm_dataset.tfrecord, 'quant_dataset.tfrecord', and 'quant_norm_dataset.tfrecord'



    # dataset_path = output_dir / 'norm_dataset.tfrecord'
    # create_dataset(n_samples, dataset_config, dataset_path, running_stats)
    # print("\nNormalized")
    # pretty_print_dataset_elements(dataset_path)

    # dataset_path = output_dir / "quant_dataset.tfrecord"
    # create_dataset(n_samples, dataset_config, dataset_path, normalization=False, quantization=True)
    # print("\nQuantized")
    # pretty_print_dataset_elements(dataset_path)

    # dataset_path = output_dir / "quant_norm_dataset.tfrecord"
    # create_dataset(n_samples, dataset_config, dataset_path, normalization=True, quantization=True)
    # print("\nQuantized and Normalized")
    # pretty_print_dataset_elements(dataset_path)


if __name__ == "__main__":
    main()
