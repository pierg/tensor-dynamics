from datetime import datetime
from pathlib import Path
import toml
from shared import config_file, results_folder, data_config_file, datasets_folder
from src.data import create_dataset
from src.datagen.dataset_utils import compute_and_print_dataset_statistics, count_elements_and_batches, get_dataset_statistics
from src.datagen.runningstats import RunningStatsDatapoints
from src.datagen.tf_dataset import load_sharded_tfrecord_dataset
from src.shared.utils import parse_arguments


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

    base_folder_path = Path(datasets_folder / f"dataset_shards_{data_config['dataset']['seed']}")


    # Check if the base folder for the dataset shards already exists.
    if not base_folder_path.exists():
        # The folder doesn't exist, so we assume the dataset hasn't been created/sharded yet.
        # Call the function to create (and shard) the dataset. This function should return
        # the path where the dataset has been saved (base_norm_path).
        create_dataset(base_folder_path, data_config)

    base_norm_path = base_folder_path / "normalized"

    # Load the sharded TFRecord dataset for training, validation, and testing.
    train_dataset, val_dataset, test_dataset = load_sharded_tfrecord_dataset(base_norm_path, data_config["dataset"])

    # Print statistics (shape, size, etc..) on train_dataset, val_dataset, test_dataset
    compute_and_print_dataset_statistics(train_dataset, "Training Dataset")
    compute_and_print_dataset_statistics(val_dataset, "Validation Dataset")
    compute_and_print_dataset_statistics(test_dataset, "Test Dataset")




if __name__ == "__main__":
    main()
