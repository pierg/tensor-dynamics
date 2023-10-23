from pathlib import Path
import pickle
from typing import List, Tuple
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import random
import os
import subprocess

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


# Function to load secrets from a file into environment variables
def load_secrets(file_path):
    print(f"Loading secrets: {file_path}")
    with open(file_path, 'r') as file:
        for line in file:
            # Clean up the line
            cleaned_line = line.strip()

            # Ignore empty lines and comments
            if cleaned_line == "" or cleaned_line.startswith('#'):
                continue

            # Check if line contains 'export ' from shell script format and remove it
            if cleaned_line.startswith('export '):
                cleaned_line = cleaned_line.replace('export ', '', 1)  # remove the first occurrence of 'export '

            # Split the line into key and value
            if '=' in cleaned_line:
                key, value = cleaned_line.split('=', 1)
                print(f"Loading {key}")
                os.environ[key] = value
            else:
                print(f"Warning: Ignoring line, missing '=': {cleaned_line}")


def git_push(commit_message="Update files", folder=None):
    """
    Pushes changes to a GitHub repository using the token-based authentication.

    Args:
    commit_message (str, optional): The commit message. Defaults to "Update files".
    folder (Path, optional): Path object representing the folder to push. If not specified, the current directory is used.
    """
    if folder is None:
        folder = Path.cwd()  # Use the current working directory if no folder is provided
    elif not folder.is_dir():
        raise ValueError(f"{folder} does not exist or is not a directory.")

    github_token = os.getenv('GITHUB_TOKEN')
    if github_token is None:
        raise ValueError("GITHUB_TOKEN is not set in the environment variables.")

    repo_url = os.getenv('GITHUB_RESULTS_REPO')
    if repo_url is None:
        raise ValueError("GITHUB_RESULTS_REPO is not set in the environment variables.")
    
    if not repo_url.startswith("https://"):
        raise ValueError("The repository URL must start with 'https://'")

    repo_url_with_token = repo_url.replace("https://", f"https://{github_token}:x-oauth-basic@")

    original_cwd = Path.cwd()  # Save the original working directory
    try:
        os.chdir(folder)  # Change to the target directory

        # Perform git operations
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push", repo_url_with_token, "main"], check=True)  # replace "main" with your target branch if different
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while pushing to GitHub: {str(e)}")
        raise  # Rethrow the exception to handle it at a higher level of your application
    finally:
        os.chdir(original_cwd)  # Ensure that you always return to the original directory


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
    data_folder: Path, n_files: int = None
) -> Tuple[List, List]:
    """
    Load data from multiple files. If the number of files is not specified, all valid files in the data folder are loaded.

    Args:
        data_folder (Path): Path to the folder containing the data files.
        n_files (int, optional): Number of files to randomly select and process. If None, all files are processed.

    Returns:
        Tuple[List, List]: Aggregated list of features and corresponding targets from all files.
    """
    # Initialize the accumulators for features and targets
    nn_data = []
    predictions = []

    # List all ".p" files in the data folder
    all_files = sorted([file for file in Path(data_folder).iterdir() if file.suffix == ".p"], key=lambda f: f.name)

    # Randomly select files if n_files is specified
    if n_files is not None:
        all_files = random.sample(all_files, min(n_files, len(all_files)))  # select n_files or all available files

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
