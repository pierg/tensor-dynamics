import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from shared import results_folder


def load_training_info(directory):
    """
    Load training information from JSON files in all subdirectories of the provided directory.
    """
    training_data = {}

    # Recursively walk through all subdirectories
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file == "training_info.json":
                json_file = os.path.join(subdir, file)

                try:
                    # If the training_info.json file exists, read it
                    with open(json_file, "r") as file:
                        data = json.load(file)
                        # Use the directory name as a key
                        # This assumes each subdirectory has a unique name
                        dir_name = os.path.basename(subdir)
                        training_data[dir_name] = data

                except Exception as e:
                    # You might want to log exceptions here or handle them some other way
                    print(f"An error occurred: {e}")

    return training_data


def plot_training_info(training_data, metric, results_dir):
    """
    Plot comparison charts for the specified metric from the training information.
    This function now plots only the top 10 configurations based on the last recorded value of the specified metric.
    """
    plt.figure(figsize=(10, 6))

    # New: Collect the final metric values and corresponding configuration names
    final_metric_values = []
    for config, data in training_data.items():
        history = data.get("training_history", {}).get("history", {})
        if metric in history:
            final_value = history[metric][-1] if history[metric] else None  # Assumes the metric is a list
            final_metric_values.append((final_value, config))

    # New: Sort configurations by the final metric value (assumes lower is better; reverse if higher is better)
    final_metric_values.sort(key=lambda x: x[0])  # Add 'reverse=True' if higher values are better

    # New: Get only the top 10 configurations
    top_10_configs = final_metric_values[:10]

    # New: Plot only the top 10 configurations
    for final_value, config in top_10_configs:
        history = training_data[config].get("training_history", {}).get("history", {})
        plt.plot(history[metric], label=f"{config} - {metric} (final: {final_value:.5f})")  # Adjust formatting as needed

    plt.title(f"Top 10 Configurations for {metric}")
    plt.ylabel(metric)
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the figure in the results directory
    plot_filename = f"top_10_comparison_{metric}.png"
    plot_path = os.path.join(results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()  # Close the plot


def compare_plots(training_data, results_dir):
    # Specify the metrics you want to compare
    metrics_to_compare = [
        "loss",
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "root_mean_squared_error"
        # Add other metrics here
    ]

    # Create and save plots for each metric
    for metric in metrics_to_compare:
        plot_training_info(training_data, metric, results_dir)


def create_summary_file(training_data, results_dir):
    """
    Create a summary text file comparing all models' configurations and performances.
    """
    summary_lines = ["Training Summary:\n", "-----------------\n"]

    # Initialize variables to store the best performance.
    best_config = None
    best_loss = float(
        "inf"
    )  # You can use any metric you prefer to judge the 'best' model

    # Search for the best performing configuration based on the desired criterion (e.g., lowest loss).
    for config, data in training_data.items():
        evaluation_results = data.get("evaluation_results", {})
        current_loss = evaluation_results.get(
            "loss", float("inf")
        )  # Ensure this matches your chosen metric

        if current_loss < best_loss:
            best_loss = current_loss
            best_config = config

    # Adding the best configuration info at the beginning of the summary
    if best_config:
        best_config_info = training_data[best_config]
        config_name = best_config_info.get("config_name", "N/A")
        timestamp = best_config_info.get("timestamp", "N/A")

        summary_lines.append(f"Best Performing Configuration: '{best_config}'\n")
        summary_lines.append(f"Config Name: {config_name}\n")
        summary_lines.append(f"Timestamp: {timestamp}\n")
        summary_lines.append(f"Lowest Loss: {best_loss}\n\n")

    # Add information for each configuration
    for config, data in training_data.items():
        config_info = data.get("config_name", "N/A")
        timestamp = data.get("timestamp", "N/A")
        evaluation_results = data.get("evaluation_results", {})

        # Add configuration header
        summary_lines.append(f"Configuration: {config}\n")
        summary_lines.append(f"Config Name: {config_info}\n")
        summary_lines.append(f"Timestamp: {timestamp}\n\n")

        # Add model evaluation results
        summary_lines.append("Evaluation Results:\n")
        for metric, value in evaluation_results.items():
            summary_lines.append(f"  - {metric}: {value}\n")
        summary_lines.append("\n")

        # Add more sections as needed (e.g., model_config, training_history, etc.)

    # Write the summary to a text file in the results directory
    summary_file_path = os.path.join(results_dir, "summary.txt")
    with open(summary_file_path, "w") as summary_file:
        summary_file.writelines(summary_lines)


def compare_results():
    # Results directory
    directory: Path = results_folder

    # Create a results directory if it doesn't exist
    comparison_dir = directory / "comparisons"

    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)

    # Load training information from JSON files in each sub-directory
    training_data = load_training_info(directory)

    compare_plots(training_data, comparison_dir)
    create_summary_file(training_data, comparison_dir)


if __name__ == "__main__":
    compare_results()
