import tensorflow as tf
import numpy as np
from pathlib import Path
import os

# Function to convert different data types to corresponding tf.train.Feature
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_example(feature, label):
    """Creates a tf.train.Example message ready to be written to a file."""
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    feature_set = {
        'feature': _bytes_feature(tf.io.serialize_tensor(feature)),
        'label': _float_feature(label),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_set))
    return example_proto.SerializeToString()


def tf_serialize_example(f, l):
    tf_string = tf.py_function(serialize_example, (f, l), tf.string)
    return tf.reshape(tf_string, ())


def write_tfrecord(features_dataset, labels_dataset, filename):
    """Writes the dataset into TFRecord file."""
    with tf.io.TFRecordWriter(filename) as writer:
        for feature, label in zip(features_dataset, labels_dataset):
            example = serialize_example(feature, label)
            writer.write(example)

def load_dataset(filename, buffer_size):
    """Loads and prepares the dataset from TFRecord file."""
    raw_dataset = tf.data.TFRecordDataset(filename)

    # Describe how to parse the example
    feature_description = {
        'feature': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.float32),
    }

    def _parse_function(example_proto):
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        parsed_example['feature'] = tf.io.decode_jpeg(parsed_example['feature'])
        return parsed_example['feature'], parsed_example['label']

    parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    parsed_dataset = parsed_dataset.shuffle(buffer_size).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return parsed_dataset


def create_dataset(num_samples, dataset_config, output_dir):
    """
    Create a dataset suitable for feeding into a neural network.
    
    Parameters:
    num_samples (int): Number of samples to generate for the dataset.
    dataset_config (dict): Configuration parameters for dataset generation.
    output_dir (Path): Directory where the dataset will be stored.
    
    Returns:
    tf.data.Dataset: A TensorFlow Dataset object ready for training.
    """
    
    print(f"Dataset configuration: {dataset_config}\n")

    # Set the seed for reproducibility
    seed_value = dataset_config.get('seed', {}).get('value', None)
    if seed_value is not None:
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)

    # Initialize the dataset containers
    features_dataset = []
    labels_dataset = []

    # Set up the TFRecord file path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    tfrecord_file = output_dir / 'dataset.tfrecord'

    with tf.io.TFRecordWriter(str(tfrecord_file)) as writer:
        for i in range(num_samples):
            # Here, replace 'generate_parameters' and 'generate_datapoint' with your data generation logic
            parameters = generate_parameters(dataset_config['parameters'])
            datapoint = generate_datapoint(parameters)

            features = datapoint["NN_eValue_Input"] * datapoint["NN_eVector_Input"]
            labels = datapoint["NN_Prediction"].flatten()

            # Convert the features and labels to the appropriate data types for serialization
            features_dataset.append(features)
            labels_dataset.append(labels)

            example = serialize_example(features, labels)
            writer.write(example)

    print(f"Data written to {tfrecord_file}")

    # Convert the datasets to TensorFlow's Dataset object after serialization
    raw_dataset = load_dataset(str(tfrecord_file), buffer_size=10000)  # buffer_size can be adjusted

    return raw_dataset


if __name__ == "__main__":
    # For example, generate data for the 'd1' configuration
    dataset = create_dataset(2, "d1")
    # Now you can use 'dataset' with TensorFlow, for example, in model training.
