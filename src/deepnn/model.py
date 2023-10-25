import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from .utils import compute_dataset_range, compute_mean_and_variance
from shared import tb_log_dir
from datetime import datetime

class NeuralNetwork:
    def __init__(
        self,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        test_dataset: tf.data.Dataset,
        configuration: dict,
        name: str,
        instance_folder
    ):
        """
        Initialize the NeuralNetwork class with training, validation, and test datasets,
        and a specific configuration from a file.

        Args:
            train_dataset (tf.data.Dataset): Training dataset.
            validation_dataset (tf.data.Dataset): Validation dataset.
            test_dataset (tf.data.Dataset): Testing dataset.
            configuration (dict): Configuration to use.
            name (str): Name of the confirguration
        """
        self.config = configuration
        self.name = name

        # Dataset attributes
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self._analyze_datasets()

        # Extracting configuration components for easier access
        self.structure_config = self.config["structure"]
        self.compile_config = self.config["compile"]
        self.training_config = self.config["training"]

        # Building and compiling the neural network model
        self.model = self._build_model()
        self._compile_model()

        self.instance_folder = instance_folder

    def _analyze_datasets(self):
        # Compute range of train_dataset val_dataset and test_dataset
        # Compute and print the range for each dataset
        self.train_range = compute_dataset_range(self.train_dataset)
        print(f"Range of target values in training dataset: {self.train_range}")
        self.mean_train, self.variance_train = compute_mean_and_variance(self.train_dataset)
        print(f"Mean of target values in training dataset: {self.mean_train}")
        print(f"Variance of target values in training dataset: {self.variance_train}")

        self.val_range = compute_dataset_range(self.validation_dataset)
        print(f"Range of target values in validation dataset: {self.val_range}")
        self.mean_val, self.variance_val = compute_mean_and_variance(self.validation_dataset)
        print(f"Mean of target values in validation dataset: {self.mean_val}")
        print(f"Variance of target values in validation dataset: {self.variance_val}")

        self.test_range = compute_dataset_range(self.test_dataset)
        print(f"Range of target values in test dataset: {self.test_range}")
        self.mean_test, self.variance_test = compute_mean_and_variance(self.test_dataset)
        print(f"Mean of target values in test dataset: {self.mean_test}")
        print(f"Variance of target values in test dataset: {self.variance_test}")



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
 
            # For each type of layer, we handle the parameters appropriately.
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
            elif layer_type == "BatchNormalization":  # Handling BatchNormalization
                prev_layer = tf.keras.layers.BatchNormalization()(prev_layer)
            elif layer_type == "Flatten":
                prev_layer = tf.keras.layers.Flatten()(prev_layer)
            elif layer_type == "Dense":
                prev_layer = tf.keras.layers.Dense(
                    units=layer_conf["units"],
                    activation=layer_conf["activation"]
                )(prev_layer)
            elif layer_type == "Dropout":  # Handling Dropout
                prev_layer = tf.keras.layers.Dropout(
                    rate=layer_conf["rate"]
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
        log_dir = self.instance_folder / f"{self.name}"
        
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


        self.model.summary()  # Display model architecture

        # Extract training parameters from configuration
        epochs = self.training_config["epochs"]

        # Train the model
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            validation_data=self.validation_dataset,
            callbacks=[tensorboard_callback]
        )

        return history

    def evaluate_model(self, verbose=0, return_dict=True):
        """
        Evaluate the neural network model with the provided testing dataset.

        Args:
            verbose (int, optional): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Defaults to 0.
            return_dict (bool, optional): If True, loss and metric results are returned as a dictionary.

        Returns:
            dict: A dictionary containing detailed loss and metric results along with additional statistics.

        Raises:
            ValueError: If the test dataset is empty.
        """
        # Check if the test dataset is empty. The cardinality method checks the number of elements in the dataset.
        if self.test_dataset.cardinality().numpy() == 0:
            raise ValueError("Test dataset is empty, evaluation cannot be performed.")

        # Evaluating the model on different datasets helps in understanding the performance and robustness of the model.

        # The training data evaluation helps understand how well the model learned the patterns in the data it was trained on.
        train_results = self.model.evaluate(x=self.train_dataset, verbose=verbose, return_dict=return_dict)
        
        # Evaluating on validation data provides insights on how the model performs on unseen data, which is crucial for understanding its generalization.
        val_results = self.model.evaluate(x=self.val_dataset, verbose=verbose, return_dict=return_dict)

        # Finally, the test data evaluation gives the most unbiased estimate of the model's real-world performance on entirely new data.
        test_results = self.model.evaluate(x=self.test_dataset, verbose=verbose, return_dict=return_dict)

        # Calculate additional statistics, such as R-squared, which is a statistical measure that represents the proportion of the variance for the dependent variable that's explained by the independent variables in a regression model.

        # The closer R-squared is to 1, the more the model explains the variation in the target variable. Conversely, a value closer to 0 indicates the model does not explain much of the variation, highlighting potential issues with the model's fit.
        R_squared_train = 1 - (train_results['loss'] / self.variance_train)
        R_squared_val = 1 - (val_results['loss'] / self.variance_val)
        R_squared_test = 1 - (test_results['loss'] / self.variance_test)

        # Organize everything in a dictionary to return. This includes both the results from .evaluate()
        # as well as any additional statistics you've calculated.
        # This comprehensive data helps in making informed decisions and evaluations about the model's performance and potential next steps.
        evaluation_results = {
            'train': {
                'results': train_results,
                'range': self.train_range,  # Range gives an idea of the spread of values, which can influence how we interpret the model's error rates.
                'mean': self.mean_train,  # Knowing the mean helps put the model's prediction errors into context.
                'variance': self.variance_train,  # Variance helps in understanding the distribution of data.
                'R_squared': R_squared_train  # Indicates how much of the target's variability is explained by the model.
            },
            'validation': {
                'results': val_results,
                'range': self.val_range,
                'mean': self.mean_val,
                'variance': self.variance_val,
                'R_squared': R_squared_val
            },
            'test': {
                'results': test_results,
                'range': self.test_range,
                'mean': self.mean_test,
                'variance': self.variance_test,
                'R_squared': R_squared_test
            }
        }

        return evaluation_results

