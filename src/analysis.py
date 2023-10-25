import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from shared import results_folder, comparisons_folder


def create_summary_file(training_data, results_dir):
    summary_lines = ["Training Summary:\n", "-----------------\n"]

    best_performance_index = None
    best_mape = float("inf")

    # Search for the best performing configuration
    for i, data in enumerate(training_data):
        current_mape = data.get("evaluation_results", {}).get("mean_absolute_percentage_error", float("inf"))

        if current_mape < best_mape:
            best_mape = current_mape
            best_performance_index = i

    # Adding the best configuration info at the beginning of the summary
    if best_performance_index is not None:
        best_config = training_data[best_performance_index]
        config_name = best_config.get("config_name", "N/A")
        timestamp = best_config.get("timestamp", "N/A")

        summary_lines.append(f"Best Performing Configuration:\n")
        summary_lines.append(f"Config Name: {config_name}\n")
        summary_lines.append(f"Timestamp: {timestamp}\n")
        summary_lines.append(f"Lowest MAPE: {best_mape}\n\n")

    # Add information for each configuration
    for data in training_data:
        config_name = data.get("config_name", "N/A")
        timestamp = data.get("timestamp", "N/A")
        evaluation_results = data.get("evaluation_results", {})

        summary_lines.append(f"Configuration: {config_name}\n")
        summary_lines.append(f"Timestamp: {timestamp}\n\n")

        # Add model evaluation results
        summary_lines.append("Evaluation Results:\n")
        for metric, value in evaluation_results.items():
            summary_lines.append(f"  - {metric}: {value}\n")
        summary_lines.append("\n")

    summary_file_path = os.path.join(results_dir, "summary.txt")
    with open(summary_file_path, "w") as summary_file:
        summary_file.writelines(summary_lines)

def plot_training_info(training_data, metric, results_dir):
    plt.figure(figsize=(10, 6))

    metric_values = []
    for i, data in enumerate(training_data):
        history = data.get("training_history", {}).get("history", {})
        metric_values.append((history.get(metric, [None])[-1], i))  # get the last value

    # sort by the metric and get the top 10
    top_configs = sorted(metric_values, key=lambda x: x[0] if x[0] is not None else float('inf'))[:10]

    for _, config_index in top_configs:
        config = training_data[config_index]
        history = config.get("training_history", {}).get("history", {})
        if metric in history:
            values = history[metric]
            if values:
                plt.plot(values, label=f"{config.get('config_name', 'N/A')} (final: {values[-1]:.5f})")

    plt.title(f"Top 10 Configurations for {metric}")
    plt.ylabel(metric)
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f"top_10_comparison_{metric}.png"
    plot_path = os.path.join(results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()

def compare_plots(training_data, results_dir):
    # Specify the metrics you want to compare
    metrics_to_compare = [
        "loss",
        "mean_squared_error",
        "val_loss",
        "val_mean_squared_error",
    ]

    # Create and save plots for each metric
    for metric in metrics_to_compare:
        plot_training_info(training_data, metric, results_dir)


def find_training_info_json(root_directory):
    """
    This function searches for all 'training_info.json' files starting from the root directory.
    :param root_directory: The starting directory to search through.
    :return: A list of file paths that point to 'training_info.json' files.
    """
    matches = []
    for root, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename == 'training_info.json':
                matches.append(os.path.join(root, filename))
    return matches


def extract_training_info(file_paths):
    training_data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            training_data.append(data)
    return training_data



def compare_results():

    training_info_files = find_training_info_json(results_folder)

    training_data = extract_training_info(training_info_files)

    if not os.path.exists(comparisons_folder):
        os.makedirs(comparisons_folder)

    create_summary_file(training_data, comparisons_folder)
    compare_plots(training_data, comparisons_folder)


if __name__ == "__main__":
    compare_results()
