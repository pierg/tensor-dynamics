import toml
from datetime import datetime

# Local module imports
from shared import (
    results_repo_folder,
    data_config_file,
    results_runs_folder,
)
from data_dynamic import DatasetGenerator
from deepnn.datasets import Datasets
from deepnn.model import NeuralNetwork
from shared.utils import git_push, save_dict_to_json_file
from utils import handle_training_exception, main_preamble

def print_dataset_statistics(datasets: Datasets):
    """
    Print statistics of the datasets.

    Args:
        datasets (Datasets): The datasets object containing train, validation, and test datasets.
    """
    print("\nDataset Statistics:")
    print(f"Training samples: {len(list(datasets.train_dataset))}")
    print(f"Validation samples: {len(list(datasets.validation_dataset))}")
    print(f"Test samples: {len(list(datasets.test_dataset))}")
    print(f"Training stats: {datasets.train_stats}\n")


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

    dataset_generator = DatasetGenerator(dataset_config["dataset"], dataset_config["parameters"])

    train_dataset, val_dataset, test_dataset = dataset_generator.create_tf_datasets()

    datasets = Datasets(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        train_stats=dataset_generator.running_stats
    )

    # Print dataset statistics
    print_dataset_statistics(datasets)

    try:
        # Set up and train the neural network
        neural_network = NeuralNetwork(
            datasets=datasets,
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


if __name__ == "__main__":
    main()
