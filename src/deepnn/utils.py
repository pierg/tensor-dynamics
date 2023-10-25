from pathlib import Path
import pickle
from typing import List, Tuple
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import os
import subprocess
from urllib.parse import urlparse, urlunparse
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
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        # If GPU list is not empty, TensorFlow has access to GPU
        print(f"Num GPUs Available: {len(gpus)}")
        print("GPU(s) available for TensorFlow:", gpus)
    else:
        # If GPU list is empty, no GPU is accessible to TensorFlow
        print("No GPU available for TensorFlow")



def clone_private_repo(repo_url, local_path):
    try:
        # Ensure the GitHub token is available
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            raise ValueError("GitHub token is not provided")

        # Parse the provided URL
        parsed_url = urlparse(repo_url)

        # Prepare the new netloc with the token
        # The format is: TOKEN@hostname (The '@' is used to separate the token from the hostname)
        new_netloc = f"{token}@{parsed_url.netloc}"

        # Construct the new URL components with the modified netloc
        new_url_components = (parsed_url.scheme, new_netloc, parsed_url.path, parsed_url.params, parsed_url.query, parsed_url.fragment)

        # Reconstruct the full URL with the token included
        url_with_token = urlunparse(new_url_components)

        # Perform the clone operation
        subprocess.run(['git', 'clone', '--depth', '1', url_with_token, str(local_path)], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while cloning the repo: {e}")
        raise e
    

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


def is_directory_empty(path):
    """
    Check if a directory is empty or does not exist.

    Args:
    - path (str): The path of the directory to check.

    Returns:
    - bool: True if the directory is empty or does not exist, False otherwise.
    """
    if not os.path.exists(path):
        return True
    return len(os.listdir(path)) == 0



def git_pull(folder=None, prefer_local=True):
    """
    Pulls the latest changes from a GitHub repository using token-based authentication.
    If there are conflicts, prefer local changes over remote ones based on the prefer_local flag.

    Args:
        folder (Path, optional): Path object representing the folder to pull. If not specified, the current directory is used.
        prefer_local (bool, optional): If true, resolve merge conflicts by preferring local changes; otherwise, prefer remote changes.
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
        print(f"Changed directory to: {folder}")

        print("Starting to pull latest changes from remote repository...")
        # Fetch the changes from the remote repository
        subprocess.run(['git', 'fetch', repo_url_with_token], check=True)

        if prefer_local:
            print("Merging changes with strategy favoring local changes...")
            subprocess.run(['git', 'merge', '-Xours', 'FETCH_HEAD'], check=True)
        else:
            print("Merging changes with strategy favoring remote changes...")
            subprocess.run(['git', 'merge', '-Xtheirs', 'FETCH_HEAD'], check=True)

        print("Operation completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while pulling from GitHub: {str(e)}")
        raise  # Rethrow the exception to handle it at a higher level of your application
    finally:
        os.chdir(original_cwd)  # Ensure that you always return to the original directory



def git_push(folder=None, commit_message="Update files"):
    """
    Pushes changes to a GitHub repository using the token-based authentication.

    Args:
    folder (Path, optional): Path object representing the folder to push. If not specified, the current directory is used.
    commit_message (str, optional): The commit message. Defaults to "Update files".
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
        print(f"Changed directory to: {folder}")

        # Check for uncommitted changes
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True)
        local_changes = result.stdout.strip() != ""

        if local_changes:
            print("Adding local changes...")
            subprocess.run(['git', 'add', '.'], check=True)

            print(f"Committing with message: {commit_message}...")
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        else:
            print("No local changes to commit.")

        print("Pushing to the remote repository...")
        push_command = ['git', 'push', repo_url_with_token, 'main']
        try:
            subprocess.run(push_command, check=True)  # Attempt to push
        except subprocess.CalledProcessError:
            # If push fails, pull with strategy to prefer local changes
            print("Push failed. Pulling latest changes from remote repository...")
            subprocess.run(['git', 'fetch', repo_url_with_token], check=True)
            print("Merging changes with strategy favoring local changes...")
            subprocess.run(['git', 'merge', '-Xours', 'FETCH_HEAD'], check=True)
            print("Pushing merged changes to the remote repository...")
            subprocess.run(push_command, check=True)  # Attempt to push again

        print("Operation completed successfully.")

    except subprocess.CalledProcessError as e:
        # Here, you might want to add logic to handle merge conflicts by pulling and preferring local changes.
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
        item["NN_eValue_Input"] * item["NN_eVector_Input"] for item in data
    ]
    predictions = [item["NN_Prediction"].flatten() for item in data]

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
        all_targets.append(targets.numpy())  # Convert targets to numpy array and store

    # Concatenate all targets into a single numpy array
    all_targets_np = np.concatenate(all_targets, axis=0)

    # Calculate the mean and variance
    mean = np.mean(all_targets_np)
    variance = np.var(all_targets_np)

    return mean, variance

def compute_dataset_range(dataset):
    """
    Compute the range of target values in a TensorFlow dataset.

    Args:
    dataset (tf.data.Dataset): The dataset containing data points and target values.

    Returns:
    float: The range of target values.
    """
    # Initialize variables to store the max and min with opposite extreme values
    max_value = float('-inf')
    min_value = float('inf')

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
    all_files = sorted([file for file in Path(data_folder).iterdir() if file.suffix == ".p"], key=lambda f: f.name)

    # If n_files is specified, select the first n_files from the sorted list
    if n_files is not None:
        all_files = all_files[:min(n_files, len(all_files))]  # select first n_files or all available files, whichever is less

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
    config_name, neural_network, history, evaluation_results, save_folder: Path, formatted_time,
    datasets_ranges, datasets_means, datasets_variances
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
        "datasets_variances": datasets_variances
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
