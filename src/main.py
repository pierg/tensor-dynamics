import sys
import toml
from pathlib import Path
import tensorflow as tf
import numpy as np
from deepnn.model import NeuralNetwork
from deepnn.utils import git_push, load_data_from_files, load_secrets, preprocess_data, save_training_info, is_directory_empty, clone_private_repo, check_tf
from shared import config_file, results_folder, secrets_path
from analysis import compare_results
import argparse
import os
import traceback
import json
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process data and configurations.")
    parser.add_argument("--data_dir", required=True, help="Path to the data directory")
    parser.add_argument("configs", nargs="*", help="Optional configurations")
    return parser.parse_args()


def validate_directory(path):
    if not path.is_dir():
        print(f"Provided path is not a directory: {path}")
        sys.exit(1)


def load_and_preprocess_data(data_folder):
    data, predictions = load_data_from_files(data_folder, 7)
    return preprocess_data(data, predictions)


def process_configuration(
    config_name: str, config: dict, data: np.ndarray, predictions: np.ndarray, instance_folder
):
    """
    Set up and process a single neural network configuration.

    Args:
    config_name (str): Name of the configuration
    config (dict): The configuration settings for the neural network.
    data (ndarray): The dataset to be used.
    predictions (ndarray): The actual values or labels.

    Returns:
    dict: The results from evaluating the neural network.
    """
    dataset_config = config["dataset"]

    # Validate split ratios
    splits = dataset_config["splits"]
    if sum(splits) != 1.0:
        raise ValueError(f"Split ratios for the configuration don't add up to 1.0")

    train_ratio, val_ratio, test_ratio = splits

    # Calculate split indices
    total_count = len(data)
    train_split = int(train_ratio * total_count)
    val_split = int(val_ratio * total_count)  # Count of validation samples

    # Data partitioning
    train_data, remaining_data = data[:train_split], data[train_split:]
    train_predictions, remaining_predictions = (
        predictions[:train_split],
        predictions[train_split:],
    )
    val_data, test_data = remaining_data[:val_split], remaining_data[val_split:]
    val_predictions, test_predictions = (
        remaining_predictions[:val_split],
        remaining_predictions[val_split:],
    )

    # Convert the datasets to TensorFlow datasets for better performance
    # For the training phase, we often shuffle and batch the data. This can be done using tf.data for efficiency.
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_data, train_predictions))
        .shuffle(buffer_size=dataset_config["shuffle_buffer_size"])
        .batch(dataset_config["batch_size"])
    )
    # Usually, the validation and test dataset isn't shuffled; only batched.
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_predictions)).batch(
        dataset_config["batch_size"]
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_data, test_predictions)
    ).batch(dataset_config["batch_size"])

    try:
        # Initialize and train the neural network
        neural_network = NeuralNetwork(
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            test_dataset=test_dataset,
            configuration=config,
            name=config_name,
            instance_folder=instance_folder
        )
        history = neural_network.train_model()

        # Evaluation and saving results
        evaluation_results = neural_network.evaluate_model()
        save_training_info(
            config_name, neural_network, history, evaluation_results, instance_folder
        )

    except Exception as e:
        error_message = str(e)
        traceback_info = traceback.format_exc()  # Capture the traceback

        # Print the error message and traceback
        print(f"Configuration {config_name} can not be compiled and trained")

        # Save the error message and traceback to a file
        error_log = {"error_message": error_message, "traceback": traceback_info}
        error_file_path = Path(results_folder) / f"{config_name}_error_log.json"
        with open(error_file_path, "w", encoding="utf-8") as error_file:
            json.dump(error_log, error_file, ensure_ascii=False, indent=4)

    return


def main():

    check_tf()

     # Load secrets from file
    if secrets_path.exists():
        load_secrets(secrets_path)

    # Check if the results_folder is empty or doesn't exist
    if is_directory_empty(results_folder):
        # Clone the private repo
        print("Directory is empty. Cloning the private repository...")
        clone_private_repo(os.getenv('GITHUB_RESULTS_REPO'), results_folder)

    args = parse_arguments()

    # Load the configurations from the TOML file
    configs = toml.load(config_file)

    # Results folder
    instance_folder = results_folder / f"{datetime.now().strftime('%m.%d-%H.%M.%S')}"

    print(args)

    config_names = args.configs
    # If no specific configurations are provided, process all configurations
    if not config_names:
        print(
            "No specific configuration names provided. Processing all configurations."
        )
        config_names = list(configs.keys())
        print(config_names)

    data_folder = Path(os.path.expanduser(args.data_dir))
    validate_directory(data_folder)

    data, predictions = load_and_preprocess_data(data_folder)

    # Process each configuration
    for config_name in config_names:
        if config_name in configs:
            config = configs[config_name]
            print("Processing configuration...")
            process_configuration(
                config_name, config, data, predictions, instance_folder
            )  # This function is the refactored part of your main function
            print("Comparing results...")
            compare_results()
            # Push to github
            print("Pushing to github...")
            git_push(folder=results_folder)

        else:
            print(f"Configuration {config_name} not found.")

    print("All configurations have been processed and results have been saved.")


if __name__ == "__main__":

    main()

    # load_secrets(secrets_path)

    # # Check if the results_folder is empty or doesn't exist
    # if is_directory_empty(results_folder):
    #     # Clone the private repo
    #     print("Directory is empty. Cloning the private repository...")
    #     clone_private_repo(os.getenv('GITHUB_RESULTS_REPO'), results_folder)

    # main()

    # git_push(folder=results_folder)
