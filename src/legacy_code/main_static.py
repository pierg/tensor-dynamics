'''
Author: Piergiuseppe Mallozzi
Date: November 2023
'''

from datetime import datetime
from pathlib import Path

import toml

from deepnn.datasets import Datasets
from deepnn.model import NeuralNetwork

# Local module imports
from shared import (
    data_config_file,
    datasets_folder,
    results_repo_folder,
    results_runs_folder,
)
from shared.utils import git_push, save_dict_to_json_file
from utils import handle_training_exception, main_preamble

from .data_static import load_all_datasets, process_datasets


def process_configuration(config_name, config, instance_config_folder):
    """
    Process each configuration: setup, train, and evaluate the neural network.

    Args:
        config_name (str): Name of the configuration.
        config (dict): The configuration dictionary.
        instance_config_folder (Path): The folder for this instance's results.
    """
    print(f"\nProcessing {config_name}...")

    # Load the dataset configuration
    dataset_config = toml.load(data_config_file).get(config["dataset"]["id"], {})

    # If you're testing with a specific dataset, uncomment the line below
    # dataset_config = toml.load(data_config_file)["dtest"]

    # Create the dataset and obtain its folder path
    base_folder_path = process_datasets(
        dataset_config, datasets_folder, overwrite=False
    )

    datasets, running_stats = load_all_datasets(
        dataset_config, base_folder_path, transformed=False
    )

    try:
        # Set up and train the neural network
        neural_network = NeuralNetwork(
            datasets=Datasets.from_dict(datasets, running_stats),
            configuration=config,
            name=config_name,
            instance_folder=instance_config_folder,
        )

        neural_network.train_model()  # Train
        neural_network.evaluate_model()  # Evaluate
        neural_network.save_model(instance_config_folder)  # Save

        # Retrieve and display results
        results = neural_network.get_results()
        save_dict_to_json_file(results, instance_config_folder / "results.json")

        return results

    except Exception as e:
        # Handle any exceptions that occurred during training/evaluation
        handle_training_exception(e, config_name, instance_config_folder)
        raise e


def main():
    """
    Main execution function.
    """
    # Load configurations and prepare the environment
    configurations = main_preamble()

    # Create a unique instance folder for this execution's results
    instance_folder = results_runs_folder / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    instance_folder.mkdir(parents=True, exist_ok=True)

    # Process each configuration
    for config_name, config in configurations.items():
        process_configuration(config_name, config, instance_folder / config_name)
        # Saving results on GitHub
        git_push(folder=results_repo_folder)

    # print("\nAnalysing and saving results...")
    # analyze_and_push(results_folder)


if __name__ == "__main__":
    main()
