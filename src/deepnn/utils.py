from pathlib import Path
import pickle
from typing import List, Tuple
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt


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

    # Extract features and targets
    nn_data = [
        item["NN_eValue_Input"][1:] * item["NN_eVector_Input"][:, 1:] for item in data
    ]
    predictions = [item["NN_Prediction"].flatten() for item in data]

    return nn_data, predictions


def load_data_from_files(
    data_folder: Path, file_indices: List[int] = None
) -> Tuple[List, List]:
    """
    Load data from multiple files. If specific file indices are not provided, all valid files in the data folder are loaded.

    Args:
        data_folder (Path): Path to the folder containing the data files.
        file_indices (List[int], optional): List of indices corresponding to the data files. If None, all files are processed.

    Returns:
        Tuple[List, List]: Aggregated list of features and corresponding targets from all files.
    """
    # Initialize the accumulators for features and targets
    nn_data = []
    predictions = []

    # If no specific files are requested, load all suitable files in the directory
    if file_indices is None:
        # List all files in the data folder
        all_files = sorted(Path(data_folder).iterdir(), key=lambda f: f.name)
        # Filter files based on your criteria (e.g., extension, format, etc.)
        data_files = [
            file
            for file in all_files
            if file.suffix == ".p"
            and file.stem.startswith("MM_Demixing_Training_Data2_")
        ]

        # Load data from each file
        for file in data_files:
            nn_data_chunk, predictions_chunk = load_data_from_file(file)
            nn_data.extend(nn_data_chunk)
            predictions.extend(predictions_chunk)
    else:
        # If specific file indices are provided, only load these files
        for index in file_indices:
            file = data_folder / f"MM_Demixing_Training_Data2_{index}.p"
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
    config_name, neural_network, history, evaluation_results, save_folder: Path
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
    }

    # Create the save directory
    dir_name = save_folder / f"{config_name}_{datetime.now().strftime('%m.%d-%H.%M.%S')}"
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