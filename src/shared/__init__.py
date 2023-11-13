'''
Author: Piergiuseppe Mallozzi
Date: November 2023
Description: Initialization for shared module, defining key directory paths for configuration, results, and datasets.
'''

from pathlib import Path

# Path to the current file (__init__.py)
this_file: Path = Path(__file__)

# Path to the main repository folder, inferred from the location of this file
main_repo_folder: Path = this_file.parent.parent.parent

# Path to the configuration folder within the main repository
config_folder: Path = main_repo_folder / "config"

# Paths to specific configuration files within the configuration folder
config_file: Path = config_folder / "configurations.toml"
data_config_file: Path = config_folder / "data.toml"

# Path to the results folder within the main repository
results_repo_folder: Path = main_repo_folder / "results"

# Specific subfolders within the results folder for runs and comparisons
results_runs_folder: Path = results_repo_folder / "runs"
comparisons_folder: Path = results_repo_folder / "comparisons"

# Path to the secrets file (usually for sensitive data like API keys or credentials)
secrets_path: Path = main_repo_folder / ".secrets"

# Path to the datasets folder within the main repository
datasets_folder: Path = main_repo_folder / "data"

# Path to the data statistics folder within the results folder
data_statistics_folder: Path = results_repo_folder / "statistics"
