'''
Author: Piergiuseppe Mallozzi
Date: November 2023
Description: Description for this file
'''

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import toml

# Imports from local modules
from dataset.dataset_generator import DatasetGenerator
from deepnn.datasets import Datasets
from deepnn.model import NeuralNetwork
from shared import data_config_file, results_repo_folder, results_runs_folder
from shared.utils import git_push, save_dict_to_json_file
from utils import handle_training_exception, main_preamble


def process_configuration(
    config_name: str,
    training_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    use_stats_of: int,
    instance_config_folder: Path,
) -> Dict[str, Any]:
    """
    Process a specific configuration: setup, train, and evaluate the neural network.

    Args:
        config_name (str): Name of the configuration.
        training_config (Dict[str, Any]): Configuration dictionary for training.
        dataset_config (Dict[str, Any]): Configuration dictionary for the dataset.
        use_stats_of (int): Parameter specifying which statistics to use.
        instance_config_folder (Path): Folder for saving results of this configuration.

    Returns:
        Dict[str, Any]: Results of the training and evaluation process.
    """
    print(f"\nProcessing {config_name}...")

    # Dataset generation
    dataset_generator = DatasetGenerator(
        dataset_config["shape"], dataset_config["parameters"], use_stats_of
    )
    train_dataset, val_dataset, test_dataset = dataset_generator.create_tf_datasets()

    # Data preparation for the neural network
    datasets = Datasets(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        train_stats=dataset_generator.running_stats,
    )

    try:
        # Neural network setup
        neural_network = NeuralNetwork(
            datasets=datasets,
            configuration=training_config,
            name=config_name,
            instance_folder=instance_config_folder,
        )

        # Saving initial configuration
        info = neural_network.get_info()
        save_dict_to_json_file(info, instance_config_folder / "info.json")

        # Model training
        neural_network.train_model()

        # Post-training actions: Save model and results
        neural_network.save_model(instance_config_folder)
        results = neural_network.get_results()
        save_dict_to_json_file(results, instance_config_folder / "final_results.json")

        return results

    except Exception as e:
        handle_training_exception(e, config_name, instance_config_folder)
        raise e


def main() -> None:
    """
    Main execution function: Load configurations, process them, and push results.
    """
    training_configs = main_preamble()
    instance_folder = results_runs_folder / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    instance_folder.mkdir(parents=True, exist_ok=True)

    dataset_configs = toml.load(data_config_file)

    for config_name, training_config in training_configs.items():
        dataset_t_config = dataset_configs[training_config["dataset"]["training"]]
        use_stats_of = training_config["dataset"].get(
            "use_stats_of", dataset_t_config["shape"]["n_samples"]
        )

        # Process each configuration
        process_configuration(
            config_name,
            training_config,
            dataset_t_config,
            use_stats_of,
            instance_folder / config_name,
        )

        # Push results to the repository
        git_push(folder=results_repo_folder)


if __name__ == "__main__":
    main()
