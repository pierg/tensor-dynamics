import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, TerminateOnNaN
from tensorflow.keras.models import Model

from deepnn.callback import CustomSaveCallback
from deepnn.datasets import Datasets
from deepnn.metrics import R_squared


class NeuralNetwork:
    def __init__(self, datasets: Datasets, configuration: dict, name: str, instance_folder: Path):
        """
        Initializes a neural network instance with the given datasets, configuration, name, and instance folder.

        Args:
            datasets (Datasets): Datasets for training, validation, and testing.
            configuration (dict): Configuration dictionary for the neural network.
            name (str): Name of the neural network configuration.
            instance_folder (Path): Folder path for saving model instances and logs.
        """
        self.config = configuration
        self.name = name
        self.instance_folder = instance_folder

        # Assign datasets
        self.datasets = datasets
        self.train_dataset = datasets.train_dataset
        self.validation_dataset = datasets.validation_dataset
        self.test_dataset = datasets.test_dataset

        # Extract specific configuration sections
        self.structure_config = self.config["structure"]
        self.compile_config = self.config["compile"]
        self.training_config = self.config["training"]

        # Timing attributes
        self.training_start_time = None
        self.time_evaluation = None
        self.time_training = None

        # Initialize the neural network model
        self.initialize_model()

    def initialize_model(self):
        """
        Initializes the neural network model with the defined architecture and compilation settings.
        """
        # Set up a distribution strategy for multi-GPU training, if available
        self.strategy = tf.distribute.MirroredStrategy()
        with self.strategy.scope():
            # Build and compile the model
            self.model = self._build_model()
            self._compile_model()

        # Early stopping callback to prevent overfitting
        self.early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)

    def _build_model(self) -> Model:
        """
        Builds the neural network model based on the structure configuration.

        Returns:
            Model: The built but uncompiled Keras model.
        """
        for sample_batch in self.train_dataset.take(1):
            # sample_batch is a tuple of (input_batch, target_batch)
            # We only need the input_batch to determine the shape
            input_batch, _ = sample_batch
            break  # We only need one batch

        input_shape = input_batch.shape[1:]

        # Initialize model layers based on configuration
        model_input = tf.keras.layers.Input(shape=input_shape, name="Input_Layer")

        # Previous layer keeps track of the current layer to add, starting with the input
        prev_layer = model_input

        # Dynamically add layers from the configuration
        for layer_conf in self.structure_config["layers"]:
            layer_type = layer_conf["type"]

            # For each type of layer, we handle the parameters appropriately.
            if layer_type == "Conv2D":
                # Reshaping here...
                prev_layer = tf.keras.layers.Conv2D(
                    filters=layer_conf["filters"],
                    kernel_size=tuple(layer_conf["kernel_size"]),
                    activation=layer_conf["activation"],
                )(prev_layer)
            elif layer_type == "MaxPooling2D":
                prev_layer = tf.keras.layers.MaxPooling2D(
                    pool_size=tuple(layer_conf["pool_size"])
                )(prev_layer)
            elif layer_type == "BatchNormalization":  # Handling BatchNormalization
                prev_layer = tf.keras.layers.BatchNormalization()(prev_layer)
            elif layer_type == "Flatten":
                prev_layer = tf.keras.layers.Flatten()(prev_layer)
            elif layer_type == "Dense":
                prev_layer = tf.keras.layers.Dense(
                    units=layer_conf["units"], activation=layer_conf["activation"]
                )(prev_layer)
            elif layer_type == "Dropout":  # Handling Dropout
                prev_layer = tf.keras.layers.Dropout(rate=layer_conf["rate"])(
                    prev_layer
                )
            else:
                raise ValueError(f"Layer type '{layer_type}' not recognized.")

        # Final model structure
        model = tf.keras.Model(inputs=model_input, outputs=prev_layer)
        return model

    def _compile_model(self):
        """
        Compiles the neural network model with the defined optimizer, loss function, and metrics.
        """
        """
        Compile the neural network model based on the configuration.
        """

        # Convert string metrics to actual TensorFlow objects and add the custom R-squared metric
        actual_metrics = []
        for metric in self.compile_config["metrics"]:
            if hasattr(tf.keras.metrics, metric):
                actual_metrics.append(getattr(tf.keras.metrics, metric)())
            else:
                if metric == "R_squared":
                    # Ensure that R_squared is a defined metric function in your program
                    actual_metrics.append(R_squared())
                else:
                    raise ValueError(f"Unknown metric: {metric}")

        # Set a default clipvalue if it's not provided
        default_clipvalue = 1.0
        clipvalue = self.compile_config.get("clipvalue", default_clipvalue)

        # Get the optimizer class name from the compile configuration
        optimizer_class_name = self.compile_config["optimizer"]

        # Create the optimizer configuration dictionary, including gradient clipping
        optimizer_config = {
            "class_name": optimizer_class_name,
            "config": {"clipvalue": clipvalue},  # Set gradient clipping by value
        }

        # Retrieve the optimizer object with the specified configuration
        optimizer = tf.keras.optimizers.get(optimizer_config)

        # Compile the model with the specified loss, metrics, and optimizer including gradient clipping
        self.model.compile(
            optimizer=optimizer,
            loss=self.compile_config["loss"],
            metrics=actual_metrics,  # Use the instantiated metrics list
            run_eagerly=self.compile_config.get(
                "run_eagerly", False
            ),  # Set run_eagerly to False by default, can be overridden
        )

    def train_model(self):
        """
        Trains the neural network model using the training dataset and validates using the validation dataset.
        """
        self.training_start_time = time.time()

        # Define the directory where TensorBoard logs will be stored
        log_dir = self.instance_folder
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Print the model summary to understand the layer architecture and parameters
        self.model.summary()

        # Retrieve the number of epochs from the training configuration
        epochs = self.training_config["epochs"]

        # Initialize the custom callback for saving the model
        save_callback = CustomSaveCallback(
            neural_network=self,  # Pass the entire NeuralNetwork instance
            interval=self.training_config.get(
                "save_interval", 5
            ),  # Get the interval from config or default to 5
        )

        # Define callbacks including TensorBoard, EarlyStopping, custom saving, and TerminateOnNaN
        callbacks_list = [
            tensorboard_callback,
            self.early_stopping,
            save_callback,
            TerminateOnNaN(),  # Callback to terminate training if NaN loss is encountered
        ]

        # Train the model using the provided datasets and callbacks
        self.history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.validation_dataset,
            callbacks=callbacks_list,
            verbose=1,  # This will print one line per epoch
        )

        # Record the end time of training
        end_time = time.time()
        # Calculate the total training time
        self.time_training = end_time - self.training_start_time

    def evaluate_model(self, verbose=1) -> dict:
        """
        Evaluates the neural network model on the training, validation, and test datasets.

        Args:
            verbose (int): Verbosity mode for evaluation.

        Returns:
            dict: A dictionary containing the evaluation results for each dataset.
        """
        if self.time_training is None:
            self.time_training = time.time() - self.training_start_time

        start_time = time.time()
        # Ensure the model has been trained before evaluation
        if self.model is None:
            raise ValueError("The model hasn't been built or trained yet.")

        # Evaluate the model on all datasets
        train_evaluation = self.model.evaluate(self.train_dataset, verbose=verbose)
        validation_evaluation = self.model.evaluate(
            self.validation_dataset, verbose=verbose
        )
        test_evaluation = self.model.evaluate(self.test_dataset, verbose=verbose)

        # Prepare a dictionary to hold all evaluation results
        self.evaluation = {
            "train": dict(zip(self.model.metrics_names, train_evaluation)),
            "validation": dict(zip(self.model.metrics_names, validation_evaluation)),
            "test": dict(zip(self.model.metrics_names, test_evaluation)),
        }

        print("Evaluation Completed")
        # Record the end time
        end_time = time.time()
        # Calculate and format the elapsed time
        self.time_evaluation = end_time - start_time

    def save_model(self, filepath: Path):
        """
        Saves the trained neural network model to the specified file path.

        Args:
            filepath (Path): The file path where the model should be saved.
        """
        # Construct the complete filepath if only a directory is provided
        filepath = filepath / "trained_model.h5"
        self.model.save(filepath)
        print(f"Model saved successfully at {filepath}")

    def get_info(self) -> dict:
        """
        Retrieves information about the neural network configuration and dataset.

        Returns:
            dict: Information about the neural network configuration and dataset.
        """
        return {
            "config_name": self.name,
            "dataset": self.datasets.to_dict(),
            "model_config": {
                "structure_config": self.structure_config,
                "compile_config": self.compile_config,
                "training_config": self.training_config,
            },
        }

    def get_results(self, interim=False) -> dict:
        """
        Retrieves the training and evaluation results of the neural network.

        Args:
            interim (bool): If True, returns interim results without training history.

        Returns:
            dict: Results including evaluation metrics and training history.
        """
        # Define common data
        results = {
            "config_name": self.name,
            "evaluation": self.evaluation,
            "training_time": self.time_training,
            "evaluation_time": self.time_evaluation,
        }

        # Add training history
        if not interim:
            results["training_history"] = self.history.history

        return results

