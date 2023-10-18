import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model


class NeuralNetwork:
    def __init__(
        self,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        configuration: dict,
    ):
        """
        Initialize the NeuralNetwork class with training, validation, and test datasets,
        and a specific configuration from a file.

        Args:
            train_dataset (tf.data.Dataset): Training dataset.
            validation_dataset (tf.data.Dataset): Validation dataset.
            test_dataset (tf.data.Dataset): Testing dataset.
            configuration (dict): Configuration to use.
        """
        self.config = configuration

        # Dataset attributes
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        # Extracting configuration components for easier access
        self.structure_config = self.config["structure"]
        self.compile_config = self.config["compile"]
        self.training_config = self.config["training"]

        # Building and compiling the neural network model
        self.model = self._build_model()
        self._compile_model()

    def _build_model(self) -> tf.keras.Model:
        """
        Build the neural network model architecture based on the configuration and input data shape.

        Returns:
            Model: Uncompiled Keras model.
        """
        # We need to extract the input shape from the training dataset.
        # Here, we'll take a single batch and infer the shape from it.
        # We assume that the first dimension is the batch size and should be discarded for the model's input shape.

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

            # For each type of layer, we handle the parameters appropriately
            # Note: Error handling and data consistency checks are recommended
            if layer_type == "Conv2D":
                prev_layer = tf.keras.layers.Conv2D(
                    filters=layer_conf["filters"],
                    kernel_size=tuple(layer_conf["kernel_size"]),
                    activation=layer_conf["activation"],
                )(prev_layer)
            elif layer_type == "MaxPooling2D":
                prev_layer = tf.keras.layers.MaxPooling2D(
                    pool_size=tuple(layer_conf["pool_size"])
                )(prev_layer)
            elif layer_type == "Flatten":
                prev_layer = tf.keras.layers.Flatten()(prev_layer)
            elif layer_type == "Dense":
                prev_layer = tf.keras.layers.Dense(
                    units=layer_conf["units"], activation=layer_conf["activation"]
                )(prev_layer)
            else:
                raise ValueError(f"Layer type '{layer_type}' not recognized.")

        # Final model structure
        model = tf.keras.Model(inputs=model_input, outputs=prev_layer)
        return model

    def _compile_model(self):
        """
        Compile the neural network model based on the configuration.
        """
        # TODO: Depending on the available optimizers and customization needed, you might want to extend this section
        self.model.compile(
            optimizer=self.compile_config["optimizer"],
            loss=self.compile_config["loss"],
            metrics=self.compile_config["metrics"],
        )

        # Assuming self.compile_config is already filled by reading the configuration file
        metrics_list = self.compile_config["metrics"]

        # Convert string metrics to actual TensorFlow objects. This is needed because the configuration file has strings
        actual_metrics = []
        for metric in metrics_list:
            if hasattr(tf.keras.metrics, metric):
                actual_metrics.append(getattr(tf.keras.metrics, metric)())
            else:
                raise ValueError(f"Unknown metric: {metric}")

        self.model.compile(
            optimizer=self.compile_config["optimizer"],
            loss=self.compile_config["loss"],
            metrics=actual_metrics,
        )

    def train_model(self):
        """
        Train the neural network model with provided datasets.
        """
        self.model.summary()  # Display model architecture

        # Extract training parameters from configuration
        epochs = self.training_config["epochs"]

        # Train the model
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.validation_dataset,  # using validation dataset
        )

        return history

    def evaluate_model(self, verbose=0, return_dict=True):
        """
        Evaluate the neural network model with the provided testing dataset.

        Args:
            verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress bar, 2 = single line. Defaults to 0.
            return_dict (bool, optional): If True, loss and metric results are returned as a dictionary.

        Returns:
            dict or list: Loss and metric results. The type of return value depends on the 'return_dict' parameter.

        Raises:
            ValueError: If the test dataset is empty.
        """
        # First, we need to check if the test dataset is empty.
        if self.test_dataset.cardinality().numpy() == 0:
            raise ValueError("Test dataset is empty, evaluation cannot be performed.")

        # Now, we proceed with the evaluation.
        results = self.model.evaluate(
            x=self.test_dataset,  # We are passing the dataset object directly.
            verbose=verbose,  # Verbosity mode, 0 or 1.
            return_dict=return_dict,  # If True, returns a dict of the results.
        )

        # If return_dict is True, results will be a dictionary with metric names as keys.
        # Otherwise, it will be a list in the order of [loss, *metrics].
        if return_dict:
            return results
        else:
            return dict(
                zip(self.model.metrics_names, results)
            )  # You can still convert it to a dict if needed.
