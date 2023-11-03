import toml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Local module imports
from shared import results_repo_folder, data_config_file, results_runs_folder
from dataset.dataset_generator import DatasetGenerator
from deepnn.datasets import Datasets
from deepnn.model import NeuralNetwork
from shared.utils import git_push
from utils import handle_training_exception, main_preamble


def process_configuration(
    config_name: str,
    training_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    use_stats_of: int,
    instance_config_folder: Path,
) -> Dict[str, Any]:
    """
    Process each configuration: setup, train, and evaluate the neural network.

    Args:
        config_name (str): Name of the configuration.
        training_config (dict): The configuration dictionary for training.
        dataset_config (dict): The configuration dictionary for dataset.
        instance_config_folder (Path): The folder for this instance's results.

    Returns:
        dict: Results of the training and evaluation.
    """
    print(f"\nProcessing {config_name}...")

    dataset_generator = DatasetGenerator(
        dataset_config["shape"], 
        dataset_config["parameters"],
        use_stats_of
    )
    train_dataset, val_dataset, test_dataset = dataset_generator.create_tf_datasets()

    datasets = Datasets(
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        test_dataset=test_dataset,
        train_stats=dataset_generator.running_stats,
    )

    try:
        neural_network = NeuralNetwork(
            datasets=datasets,
            configuration=training_config,
            name=config_name,
            instance_folder=instance_config_folder,
        )

        neural_network.train_model()  # Training now includes the save callback

        # After training is complete, save the final model and results
        neural_network.save_model(instance_config_folder)
        results = neural_network.get_results()

        # Save final results to JSON file
        results_path = instance_config_folder / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f)
        print(f"Final results saved to {results_path}")

        return results

    except Exception as e:
        handle_training_exception(e, config_name, instance_config_folder)
        raise e




def main() -> None:
    """
    Main execution function.
    """
    training_configs = main_preamble()
    instance_folder = results_runs_folder / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    instance_folder.mkdir(parents=True, exist_ok=True)

    dataset_configs = toml.load(data_config_file)

    for config_name, training_config in training_configs.items():
        dataset_t_config = dataset_configs[training_config["dataset"]["training"]]
        use_stats_of = training_config["dataset"].get("use_stats_of", dataset_t_config["shape"]["n_samples"])

        process_configuration(
            config_name, 
            training_config, 
            dataset_t_config,
            use_stats_of,
            instance_folder / config_name
        )

        git_push(folder=results_repo_folder)


if __name__ == "__main__":
    main()
