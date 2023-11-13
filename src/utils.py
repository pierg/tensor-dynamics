'''
Author: Piergiuseppe Mallozzi
Date: November 2023
Description: Description for this file
'''

import json
import logging
import os
import traceback
from pathlib import Path
from typing import Dict, List

import tensorflow as tf
import toml

from shared import config_file, results_repo_folder, secrets_path
from shared.utils import (
    clone_private_repo,
    is_directory_empty,
    load_secrets,
    parse_arguments,
)

# Configure logging for better monitoring and debugging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_configurations(file_path: str) -> Dict:
    """
    Load the configurations from a TOML file.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        Dict: A dictionary of configurations loaded from the file.
    """
    try:
        with open(file_path, "r") as file:
            return toml.load(file)
    except Exception as e:
        logging.error(f"Failed to load configurations: {e}", exc_info=True)
        return {}


def check_tf_availability() -> None:
    """
    Check TensorFlow version and GPU availability. This function is useful for
    validating the TensorFlow installation and available hardware.
    """
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"Num GPUs Available: {len(gpus)}")
        print("GPU(s) available for TensorFlow:", gpus)
    else:
        print("No GPU available for TensorFlow")


def prepare_environment() -> None:
    """
    Prepare the execution environment by checking TensorFlow, loading secrets,
    and setting up the results directory.
    """
    check_tf_availability()

    if not secrets_path.exists():
        logging.warning(f"Secrets file not found at {secrets_path}")
    else:
        load_secrets(secrets_path)

    if is_directory_empty(results_repo_folder):
        logging.info("Cloning the private repository as directory is empty...")
        repo_url = os.getenv("GITHUB_RESULTS_REPO")
        if repo_url:
            clone_private_repo(repo_url, results_repo_folder)
        else:
            logging.error("Environment variable GITHUB_RESULTS_REPO is not set.")


def filter_configurations(all_configs: Dict, specified_configs: List[str]) -> Dict:
    """
    Filter configurations based on a specified list. If no configurations are specified,
    all available configurations are returned.

    Args:
        all_configs (Dict): All available configurations.
        specified_configs (List[str]): List of configuration names to filter.

    Returns:
        Dict: Filtered configurations.
    """
    if not specified_configs:
        logging.info("Processing all configurations.")
        return all_configs

    selected_configs = {
        name: all_configs[name] for name in specified_configs if name in all_configs
    }
    non_existent_configs = set(specified_configs) - set(selected_configs.keys())
    if non_existent_configs:
        logging.warning(f'Ignored non-existent configurations: {", ".join(non_existent_configs)}')

    return selected_configs


def main_preamble() -> Dict:
    """
    Execute initial setup steps before main processing begins. This includes
    environment preparation and determining configurations to process.

    Returns:
        Dict: Configurations to be processed.
    """
    all_configs = load_configurations(config_file)
    if not all_configs:
        logging.warning("No configurations loaded. Ending process.")
        return {}

    prepare_environment()
    args = parse_arguments()
    specified_config_names = args.configs if "configs" in args else []
    return filter_configurations(all_configs, specified_config_names)


def handle_training_exception(
    e: Exception, config_name: str, results_folder: Path
) -> None:
    """
    Handle exceptions during the training process, logging error details
    and saving them to a file in the results folder.

    Args:
        e (Exception): The exception encountered during training.
        config_name (str): Name of the configuration being processed.
        results_folder (Path): Path to the folder for saving error logs.
    """
    logging.error(f"Error in configuration {config_name}: {e}", exc_info=True)
    error_log = {"error_message": str(e), "traceback": traceback.format_exc()}
    error_file_path = results_folder / f"{config_name}_error_log.json"

    try:
        with open(error_file_path, "w", encoding="utf-8") as error_file:
            json.dump(error_log, error_file, ensure_ascii=False, indent=4)
    except IOError as e:
        logging.error(f"Failed to write error log: {e}", exc_info=True)
