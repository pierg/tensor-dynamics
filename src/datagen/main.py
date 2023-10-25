from pathlib import Path
import toml
import tensorflow as tf
import numpy as np
import os
from src.datagen.michaelis import *
from src.datagen.dataset import *

def main():
    # Configuration
    num_samples = 100
    dataset_config = {
        'seed': {'value': None},  # We will set this later
        'parameters': {'param_size': 10}  # Just an example parameter
    }
    output_dir = Path("./dataset_dir")

    # Step 1: Generate with a random seed
    random_seed = np.random.randint(0, 10000)  # for example between 0 and 10000
    print(f"Random seed for dataset generation: {random_seed}")
    dataset_config['seed']['value'] = random_seed

    # Store the seed for later comparison
    seed_storage_path = output_dir / "seed_value.txt"
    with open(seed_storage_path, 'w') as seed_file:
        seed_file.write(str(random_seed))

    # Generate and store the dataset
    create_dataset(num_samples, dataset_config, output_dir)

    # Step 2: Load the stored dataset
    loaded_dataset = load_dataset(str(output_dir / 'dataset.tfrecord'), buffer_size=10000)

    # Step 3: Regenerate with the same seed
    with open(seed_storage_path, 'r') as seed_file:
        stored_seed = int(seed_file.read().strip())
    
    print(f"Using stored seed for regeneration: {stored_seed}")
    dataset_config['seed']['value'] = stored_seed

    regenerated_dataset = create_dataset(num_samples, dataset_config, output_dir / "regenerated")

    # Step 4: Compare the datasets
    # This can be complex, depending on what "comparison" means in context.
    # For tensors or complex structures, you might need more sophisticated metric-based comparison.
    for (original_features, original_labels), (regenerated_features, regenerated_labels) in zip(loaded_dataset, regenerated_dataset):
        comparison_result = tf.reduce_all(tf.equal(original_features, regenerated_features)).numpy()
        print(f"Are the datasets identical? {'Yes' if comparison_result else 'No'}")

        # If they are not identical, we should stop the comparison to investigate.
        if not comparison_result:
            break
