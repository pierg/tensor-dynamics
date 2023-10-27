import os
import json
import matplotlib.pyplot as plt
from results.analysis import extract_training_info, find_training_info_json
from src.shared.utils import git_pull, git_push

# Separate functions for better modularization and readability

def find_best_configuration(training_data):
    """
    Identifies the configuration with the best performance based on MAPE.
    """
    best_performance_index = None
    best_mape = float("inf")

    for i, data in enumerate(training_data):
        current_mape = data.get("evaluation_results", {}).get(
            "mean_absolute_percentage_error", float("inf")
        )

        if current_mape < best_mape:
            best_mape = current_mape
            best_performance_index = i

    return best_performance_index, best_mape

def generate_summary_content(training_data):
    """
    Generates summary text content based on training data.
    """
    summary_lines = ["Training Summary:\n", "-----------------\n"]

    best_index, best_mape = find_best_configuration(training_data)

    # Handle the best configuration summary
    if best_index is not None:
        best_config = training_data[best_index]
        summary_lines += format_best_config_summary(best_config, best_mape)

    # Handle individual configuration summaries
    for data in training_data:
        summary_lines += format_config_summary(data)

    return summary_lines

def format_best_config_summary(config, best_mape):
    """
    Formats the summary for the best configuration.
    """
    lines = [
        f"Best Performing Configuration:\n",
        f"Config Name: {config.get('config_name', 'N/A')}\n",
        f"Timestamp: {config.get('timestamp', 'N/A')}\n",
        f"Lowest MAPE: {best_mape}\n\n"
    ]
    return lines

def format_config_summary(config):
    """
    Formats the summary for an individual configuration.
    """
    lines = [
        f"Configuration: {config.get('config_name', 'N/A')}\n",
        f"Timestamp: {config.get('timestamp', 'N/A')}\n\n",
        "Evaluation Results:\n"
    ]
    for metric, value in config.get("evaluation_results", {}).items():
        lines.append(f"  - {metric}: {value}\n")
    lines.append("\n")
    return lines

def save_summary_file(summary_content, directory):
    """
    Saves the summary file to the specified directory.
    """
    summary_file_path = os.path.join(directory, "summary.txt")
    with open(summary_file_path, "w") as file:
        file.writelines(summary_content)

def plot_training_info(training_data, metric, results_dir):
    plt.figure(figsize=(10, 6))

    metric_values = []
    for i, data in enumerate(training_data):
        history = data.get("training_history", {}).get("history", {})
        metric_values.append((history.get(metric, [None])[-1], i))  # get the last value

    # sort by the metric and get the top 10
    top_configs = sorted(
        metric_values, key=lambda x: x[0] if x[0] is not None else float("inf")
    )[:10]

    for _, config_index in top_configs:
        config = training_data[config_index]
        history = config.get("training_history", {}).get("history", {})
        if metric in history:
            values = history[metric]
            if values:
                plt.plot(
                    values,
                    label=f"{config.get('config_name', 'N/A')} (final: {values[-1]:.5f})",
                )

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


def compare_results(results_folder):
    """
    Main function to handle the comparison of results and generation of summaries and plots.
    """
    training_info_files = find_training_info_json(results_folder)
    training_data = extract_training_info(training_info_files)

    comparisons_folder = results_folder / "comparisons"

    if not os.path.exists(comparisons_folder):
        os.makedirs(comparisons_folder)

    summary_content = generate_summary_content(training_data)
    save_summary_file(summary_content, comparisons_folder)

    compare_plots(training_data, comparisons_folder)


def push_results(results_folder):
    """
    Main function for analysis and pushing the results to the repository.
    """
    git_pull(folder=results_folder, prefer_local=True)
    git_push(folder=results_folder)


def analyze_and_push(results_folder):
    """
    Main function for analysis and pushing the results to the repository.
    """
    git_pull(folder=results_folder, prefer_local=True)
    print("Comparing results...")
    compare_results(results_folder)
    print("Pushing to GitHub, overwriting remote comparisons...")
    git_push(folder=results_folder)

if __name__ == "__main__":
    from shared import results_folder
    analyze_and_push(results_folder=results_folder)
