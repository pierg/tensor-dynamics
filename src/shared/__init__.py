from pathlib import Path

# Path to the current file
this_file: Path = Path(__file__)

# Path to the main repo folder
main_repo_folder: Path = this_file.parent.parent.parent

# Path to configuration folder
config_folder: Path = main_repo_folder / "config"

# Path to configuration file
config_file: Path = config_folder / "configurations.toml"
data_config_file: Path = config_folder / "data.toml"
data_statistics_folder: Path = config_folder / "statistics"

# Path to results folder
results_repo_folder: Path = main_repo_folder / "results"
results_runs_folder: Path = results_repo_folder / "runs"
comparisons_folder: Path = results_repo_folder / "comparisons"

# Seretes path
secrets_path: Path = main_repo_folder / ".secrets"

# Datasets folder path
datasets_folder: Path = main_repo_folder / "data"
