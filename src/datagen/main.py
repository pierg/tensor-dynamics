import toml
import tensorflow as tf
import numpy as np

from src.datagen.utils import *

def main(config_path, dataset_key):
    # Load and read the configuration file
    config = toml.load(config_path)

    # Extract the specific dataset configuration based on the provided key
    dataset_config = config[dataset_key]

    # Extract parameters and seed from the specific configuration
    seed_value = dataset_config['seed']['value']
    param_config = dataset_config['parameters']

    # Set the seed for reproducibility
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Define the parameter ranges based on the config
    kinetic_coeff_range = {
        'k1': (param_config['k1_min'], param_config['k1_max']), 
        'k2': (param_config['k2_min'], param_config['k2_max']), 
        'k3': (param_config['k3_min'], param_config['k3_max'])
    }

    initial_concentration_range = {
        'E0': (param_config['E0_min'], param_config['E0_max']), 
        'S0': (param_config['S0_min'], param_config['S0_max'])  # ES0 and P0 are assumed to be zero
    }

    spectra_config = (param_config['spectra_min'], param_config['spectra_max'])
    amplitude_config = (param_config['amplitude_min'], param_config['amplitude_max'])

    # Placeholder for generated data
    features_dataset = []
    labels_dataset = []

    # Simulate the generation of multiple datapoints (you can decide the number)
    for _ in range(100):  # or however many datapoints you need
        parameters = generate_parameters(kinetic_coeff_range, initial_concentration_range, spectra_config, amplitude_config)
        datapoint = generate_datapoint(parameters)

        # Extract features and labels from the generated datapoint
        features = datapoint["NN_eValue_Input"][1:] * datapoint["NN_eVector_Input"][:, 1:]
        label = datapoint["NN_Prediction"].flatten()

        features_dataset.append(features)
        labels_dataset.append(label)

    # Convert to TensorFlow's Dataset object, optimized for TensorFlow operations
    tf_dataset = tf.data.Dataset.from_tensor_slices((features_dataset, labels_dataset))

    # Apply dataset transformations like batching, prefetching, etc.
    tf_dataset = tf_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    return tf_dataset

if __name__ == "__main__":
    # For example, generate data for the 'd1' configuration
    dataset = main("data.toml", "d1")
    # Now you can use 'dataset' with TensorFlow, for example, in model training.
