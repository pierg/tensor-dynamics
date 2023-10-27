import os
import json
import toml
import logging
import traceback
from typing import List, Dict
from pathlib import Path
from shared import config_file, secrets_path, results_folder
from src.shared.utils import (
    clone_private_repo, 
    is_directory_empty, 
    load_secrets, 
    parse_arguments,
)
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_configurations(file_path: str) -> Dict:
    """
    Load the configurations from the specified TOML file.

    :param file_path: The path to the configuration file.
    :return: A dictionary containing the configurations.
    """
    try:
        with open(file_path, 'r') as file:
            config = toml.load(file)
            return config
    except Exception as e:
        logging.error(f'Failed to load configurations: {e}', exc_info=True)
        return {}


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


def prepare_environment() -> None:
    """
    Prepare the necessary environment, including checking TensorFlow,
    loading secrets, and setting up the results directory.
    """
    check_tf()

    if not secrets_path.exists():
        logging.warning(f'Secrets file not found at {secrets_path}')
    else:
        load_secrets(secrets_path)

    if is_directory_empty(results_folder):
        logging.info('Directory is empty. Cloning the private repository...')
        repo_url = os.getenv('GITHUB_RESULTS_REPO')
        if repo_url:
            clone_private_repo(repo_url, results_folder)
        else:
            logging.error('Environment variable GITHUB_RESULTS_REPO is not set.')


def filter_configurations(all_configs: Dict, specified_configs: List[str]) -> Dict:
    """
    Filter the configurations based on the specified list. If the list is empty,
    all configurations will be used.

    :param all_configs: All available configurations.
    :param specified_configs: The list of specified configuration names.
    :return: A dictionary containing the configurations to be processed.
    """
    if not specified_configs:
        logging.info('No specific configuration names provided. Processing all configurations.')
        return all_configs

    selected_configs = {name: all_configs[name] for name in specified_configs if name in all_configs}
    non_existent_configs = set(specified_configs) - set(selected_configs.keys())

    if non_existent_configs:
        logging.warning(f'Non-existent configurations ignored: {", ".join(non_existent_configs)}')

    return selected_configs


def main_preamble() -> Dict:
    """
    Set up the environment and determine the configurations to process.

    :return: A dictionary containing configurations to be processed.
    """
    all_configs = load_configurations(config_file)

    if not all_configs:
        logging.warning('No configurations loaded, ending process.')
        return {}

    prepare_environment()

    args = parse_arguments()
    specified_config_names = args.configs if 'configs' in args else []

    return filter_configurations(all_configs, specified_config_names)


def handle_training_exception(e: Exception, config_name: str, results_folder: Path) -> None:
    """
    Handle exceptions that occur during the training process.

    :param e: The exception object.
    :param config_name: The name of the neural network configuration.
    :param results_folder: The directory path to save error logs.
    """
    logging.error(f'Error during processing configuration {config_name}: {e}', exc_info=True)

    # Record the traceback information along with the error message
    error_log = {
        'error_message': str(e),
        'traceback': traceback.format_exc()
    }

    error_file_path = results_folder / f'{config_name}_error_log.json'
    
    try:
        with open(error_file_path, 'w', encoding='utf-8') as error_file:
            json.dump(error_log, error_file, ensure_ascii=False, indent=4)
    except IOError as e:
        logging.error(f'Failed to write error log: {e}', exc_info=True)


