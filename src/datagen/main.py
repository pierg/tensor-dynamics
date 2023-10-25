import numpy as np
import tensorflow as tf
from pathlib import Path
from src.datagen.dataset import *

import numpy as np
import tensorflow as tf

def are_datasets_equal(dataset1: tf.data.Dataset, dataset2: tf.data.Dataset) -> bool:
    """
    Compares two TensorFlow datasets to check if they are practically identical,
    allowing for small differences in floating-point calculations.

    :param dataset1: First dataset.
    :param dataset2: Second dataset.
    :return: True if practically identical, False otherwise.
    """
    for (original_elements, original_labels), (regenerated_elements, regenerated_labels) in zip(dataset1, dataset2):
        # Ensure data points are numpy arrays for comparison
        original_data_elements = original_elements.numpy() if isinstance(original_elements, tf.Tensor) else original_elements
        regenerated_data_elements = regenerated_elements.numpy() if isinstance(regenerated_elements, tf.Tensor) else regenerated_elements

        original_data_labels = original_labels.numpy() if isinstance(original_labels, tf.Tensor) else original_labels
        regenerated_data_labels = regenerated_labels.numpy() if isinstance(regenerated_labels, tf.Tensor) else regenerated_labels

        # Compare the data points using a tolerance-based comparison for the elements and the labels
        if not (np.allclose(original_data_elements, regenerated_data_elements, rtol=1e-05, atol=1e-08) and 
                np.allclose(original_data_labels, regenerated_data_labels, rtol=1e-05, atol=1e-08)):
            return False
    return True



def main() -> None:

    # Set paths and load configurations
    current_dir = Path(__file__).resolve().parent
    data_config_file = current_dir / "data.toml"
    data_config = toml.load(data_config_file)
    dataset_config = data_config["d1"]  # Assuming "d1" is a placeholder for your actual config key


    seed_value = dataset_config.get('dataset', {}).get('seed', None)
    n_samples = dataset_config.get('dataset', {}).get('n_samples', None)
    batch_size = dataset_config.get('dataset', {}).get('batch_size', None)


    output_dir = Path(__file__).parent.resolve() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)


    # Step 1: Using seed
    print(f"Random seed for dataset generation: {seed_value}")

    # Generate and store the dataset.
    dataset_path = output_dir / 'dataset.tfrecord'
    create_dataset(n_samples, dataset_config, dataset_path)

    pretty_print_dataset_elements(dataset_path)

    # Step 2: Load the stored dataset.
    loaded_dataset = load_dataset(dataset_path, batch_size)

    # Step 3: Regenerate with the same seed.
    print(f"Using stored seed for regeneration: {seed_value}")

    regenerated_dataset_path = output_dir / "regenerated_dataset.tfrecord"
    create_dataset(n_samples, dataset_config, regenerated_dataset_path)

    # pretty_print_dataset_elements(regenerated_dataset_path)

    # Reload the regenerated dataset for comparison
    regenerated_dataset = load_dataset(regenerated_dataset_path, batch_size)

    # Step 4: Compare the datasets.
    are_identical = are_datasets_equal(loaded_dataset, regenerated_dataset)
    print(f"Are the datasets identical? {'Yes' if are_identical else 'No'}")

if __name__ == "__main__":
    main()
