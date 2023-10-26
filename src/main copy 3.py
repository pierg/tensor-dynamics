from pathlib import Path
import toml
from shared import config_file, results_folder, data_config_file, datasets_folder
from src.datagen.dataset_utils import pretty_print_dataset_elements, pretty_print_sharded_dataset_elements
from src.datagen.runningstats import RunningStatsDatapoints
from src.datagen.tf_dataset import create_sharded_tfrecord, transform_and_save_sharded_dataset
from src.shared.utils import parse_arguments
import datetime

def main():
    
    # Load the configurations from the TOML file
    configs = toml.load(config_file)

    # Load the configurations from the TOML file
    data_config = toml.load(data_config_file)["d1"]

    args = parse_arguments()

    # Results folder
    instance_folder = results_folder / f"{datetime.now().strftime('%m.%d-%H.%M.%S')}"

    config_names = args.configs
    
    # If no specific configurations are provided, process all configurations
    if not config_names:
        print(
            "No specific configuration names provided. Processing all configurations."
        )
        config_names = list(configs.keys())
        print(config_names)
    else:
        print(f"Processing configurations: {config_names}")


    base_file_path = Path(datasets_folder / f"dataset_shards_{data_config['dataset']['seed']}")
    base_file_path.mkdir(exist_ok=True)

    base_origin_path = base_file_path / "origin"
    base_origin_path.mkdir(exist_ok=True)

    shard_prefix = "shard"
    num_shards = data_config['dataset']['n_shards']
                                 

    running_stats = RunningStatsDatapoints()
    create_sharded_tfrecord(base_origin_path, data_config['dataset'], running_stats)

    print("\nOriginal Dataset:")
    # This function is assumed to be defined to handle the printing of elements from sharded datasets.
    pretty_print_sharded_dataset_elements(
        shards_directory=base_origin_path,
        shard_prefix=shard_prefix,
        num_shards=num_shards,
        running_stats=running_stats,  # Assuming you want to use the same stats
        n=5  # for instance, if you want to print 5 examples from each shard
    )

    base_norm_path = base_file_path / "normalized"
    base_norm_path.mkdir(exist_ok=True)

    # Function to transform and save the dataset, assumed to be defined. 
    # It should handle normalization and saving of sharded datasets.
    transform_and_save_sharded_dataset(
        base_origin_path,
        running_stats.get_feature_stats(),
        running_stats.get_label_stats(),
        output_shards_directory=base_norm_path,
        shard_prefix=shard_prefix,
        num_shards=num_shards,
    )

    print("\nNormalized Dataset:")
    # Assuming running_stats is updated during the transformation and you want to print based on the updated stats.
    pretty_print_sharded_dataset_elements(
        shards_directory=norm_dataset_path,
        shard_prefix=shard_prefix,  # if the prefix is the same
        num_shards=num_shards,
        running_stats=running_stats,  # if you want to use updated stats
        n=5  # for instance, if you want to print 5 examples from each shard
    )




    # Load the sharded TFRecord dataset for training, validation, and testing.
    split_ratios = tuple(dataset_config["splits"])
    train_dataset, val_dataset, test_dataset = load_sharded_tfrecord_dataset(base_file_path, num_shards, split_ratios, batch_size)

    # ... [Training logic]

if __name__ == "__main__":
    main()
