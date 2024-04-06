"""
Author: Piergiuseppe Mallozzi
Date: November 2023
"""

import json
import os
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse, urlunparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def print_data_info(data: np.ndarray, label: str):
    """
    Display information about the provided data.

    Args:
        data (np.ndarray): Data to display information about.
        label (str): Label to be used in print statements (e.g. "Data" or "Predictions").
    """
    print(f"{label} type: {type(data)}")
    print(f"{label} shape: {data.shape}")
    print(f"{label} size: {data.size}")
    print(f"{label} dtype: {data.dtype}")
    print("-" * 50)


def check_tf():
    print("Checking Tensorflow...")

    # Check TensorFlow version
    print("TensorFlow version: ", tf.__version__)

    # List all available GPUs
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        # If GPU list is not empty, TensorFlow has access to GPU
        print(f"Num GPUs Available: {len(gpus)}")
        print("GPU(s) available for TensorFlow:", gpus)
    else:
        # If GPU list is empty, no GPU is accessible to TensorFlow
        print("No GPU available for TensorFlow")


def load_data_from_file(filepath: Path) -> Tuple[List, List]:
    """
    Load data from the given pickle file and extract the required features and targets.

    Args:
        filepath (Path): Path to the pickle file.

    Returns:
        Tuple[List, List]: List of features and corresponding targets.
    """
    print(f"Loading data from file: {filepath}")

    # Load data from the file
    with filepath.open("rb") as f:
        data = pickle.load(f)

    print("~~~~~~~")
    print(f"NN_eValue_Input shape: {data[0]['NN_eValue_Input'].shape}")
    print(f"NN_eVector_Input shape: {data[0]['NN_eVector_Input'].shape}")
    print(f"NN_Prediction shape: {data[0]['NN_Prediction'].shape}")

    # Extract features and targets
    nn_data = [item["NN_eValue_Input"] * item["NN_eVector_Input"] for item in data]
    predictions = [item["NN_Prediction"].flatten() for item in data]

    print(f"nn_data shape: {nn_data[0].shape}")
    print(f"predictions shape: {predictions[0].shape}")
    print("~~~~~~~")

    return nn_data, predictions


def compute_mean_and_variance(dataset):
    """
    Compute the mean and variance of target values in a TensorFlow dataset.

    Args:
    dataset (tf.data.Dataset): The dataset containing data points and target values.

    Returns:
    tuple: The mean and variance of target values.
    """
    # List to store all targets
    all_targets = []

    # Iterate over the batches in the dataset
    for features, targets in dataset:
        all_targets.append(targets.numpy())

    # Concatenate all targets into a single numpy array
    all_targets_np = np.concatenate(all_targets, axis=0)

    # Calculate the mean and variance
    mean = np.mean(all_targets_np)
    mean_abs = np.mean(np.absolute(all_targets_np))
    variance = np.var(all_targets_np)

    return mean_abs, variance


def compute_dataset_range(dataset):
    """
    Compute the range of target values in a TensorFlow dataset.

    Args:
    dataset (tf.data.Dataset): The dataset containing data points and target values.

    Returns:
    float: The range of target values.
    """
    # Initialize variables to store the max and min with opposite extreme values
    max_value = float("-inf")
    min_value = float("inf")

    # Iterate over the batches in the dataset
    for features, targets in dataset:
        # Convert targets to numpy array
        targets_numpy = targets.numpy()

        # Update max and min
        max_value = max(max_value, np.max(targets_numpy))
        min_value = min(min_value, np.min(targets_numpy))

    # Calculate the range
    return max_value - min_value


def load_data_from_files(data_folder: Path, n_files: int = None) -> Tuple[List, List]:
    """
    Load data from multiple files. If the number of files is not specified, all valid files in the data folder are loaded.

    Args:
        data_folder (Path): Path to the folder containing the data files.
        n_files (int, optional): Number of files to select and process from the beginning. If None, all files are processed.

    Returns:
        Tuple[List, List]: Aggregated list of features and corresponding targets from all files.
    """
    # Initialize the accumulators for features and targets
    nn_data = []
    predictions = []

    # List all ".p" files in the data folder and sort them by name
    all_files = sorted(
        [file for file in Path(data_folder).iterdir() if file.suffix == ".p"],
        key=lambda f: f.name,
    )

    # If n_files is specified, select the first n_files from the sorted list
    if n_files is not None:
        all_files = all_files[
            : min(n_files, len(all_files))
        ]  # select first n_files or all available files, whichever is less

    # Load data from each file
    for file in all_files:
        # load_data_from_file is a hypothetical function you should replace with your actual data loading logic
        nn_data_chunk, predictions_chunk = load_data_from_file(file)
        nn_data.extend(nn_data_chunk)
        predictions.extend(predictions_chunk)

    return nn_data, predictions


def preprocess_data(data: list, predictions: list) -> tuple:
    """
    Convert data and predictions to numpy arrays and reshape as necessary.

    Args:
        data (list): Raw data loaded from files.
        predictions (list): Raw predictions loaded from files.

    Returns:
        tuple: Processed data and predictions.
    """
    print(f"BEFORE data shape: {data[0].shape}")
    print(f"BEFORE predictions shape: {predictions[0].shape}")

    # Convert list of data into a numpy array for efficient manipulation.
    data = np.array(data)

    # Reshape the data array to 4D, as required by CNNs.
    # The CNNs require data in the shape (num_samples, height, width, channels).
    # Even if data is grayscale (one channel), we need to include the single channel
    # dimension explicitly. Here, we reshape to add a single channel. This is necessary
    # to meet the input requirements of most deep learning frameworks and is also good practice
    # for maintaining consistency in data dimensions, clarity in code, and ensuring smooth
    # adaptability for potential future use with multi-channel data.
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)

    # Convert list of predictions into a numpy array for subsequent processing.
    predictions = np.array(predictions)

    print_data_info(data, "data")
    print_data_info(predictions, "predictions")

    return data, predictions


def save_training_info(
    config_name,
    neural_network,
    history,
    evaluation_results,
    save_folder: Path,
    formatted_time,
    datasets_ranges,
    datasets_means,
    datasets_variances,
):
    """
    Save training information and generate plots for the training history.

    Args:
        config_name (str): Name of the configuration
        neural_network (NeuralNetwork): NeuralNetwork object after training
        history (History): Returned history object from model training.
        evaluation_results (dict): The dictionary containing scores from the model evaluation.
        save_folder (Path): Folder where to save the info
    """
    # Create a dictionary to store all training information
    training_info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time_elapsed": formatted_time,
        "config_name": config_name,
        "model_config": {
            "structure_config": neural_network.structure_config,
            "compile_config": neural_network.compile_config,
            "training_config": neural_network.training_config,
        },
        "training_history": {
            "epochs": history.epoch,
            "history": history.history,
        },
        "evaluation_results": evaluation_results,
        "datasets_ranges": datasets_ranges,
        "datasets_means": datasets_means,
        "datasets_variances": datasets_variances,
    }

    # Create the save directory
    dir_name = save_folder / config_name
    os.makedirs(dir_name, exist_ok=True)

    # Save all training information to a JSON file
    with open(dir_name / "training_info.json", "w", encoding="utf-8") as file:
        json.dump(training_info, file, ensure_ascii=False, indent=4)

    # Saving the model architecture and weights
    neural_network.model.save(os.path.join(dir_name, "complete_model"))

    # Save all training information to a JSON file
    with open(
        os.path.join(dir_name, "training_info.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(training_info, file, ensure_ascii=False, indent=4)

    # Save the model architecture
    model_json = neural_network.model.to_json()
    with open(os.path.join(dir_name, "model_architecture.json"), "w") as json_file:
        json_file.write(model_json)

    # Plotting the training history
    # We will create plots for loss and each metric in the training history.
    for metric in ["loss"] + list(evaluation_results.keys()):
        plt.figure(figsize=(10, 6))

        # Plot training metric
        training_metric = history.history[metric]
        plt.plot(training_metric, label=f"Training {metric}")

        # If available, plot validation metric
        val_metric = history.history.get(f"val_{metric}")
        if val_metric:
            plt.plot(val_metric, label=f"Validation {metric}")

        plt.title(f"{metric} over epochs")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.grid()

        # Save the figure
        plt.savefig(
            f"{dir_name}/{metric}_over_epochs.png"
        )  # choose your desired file format and path
        plt.close()  # Close the figure to free up memory

    print(f"Training information and plots are saved in the directory: {dir_name}")
