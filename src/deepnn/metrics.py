"""
Author: Piergiuseppe Mallozzi
Date: November 2023
Description: Custom TensorFlow metrics implementation, including the R_squared metric class for model performance evaluation.
"""

import tensorflow as tf


class R_squared(tf.keras.metrics.Metric):
    """
    Custom implementation of the R-squared (coefficient of determination) metric in TensorFlow.
    This metric provides a measure of how well observed outcomes are replicated by the model.
    """

    def __init__(self, name="r_squared", **kwargs):
        """
        Initializes the R_squared metric instance.

        Args:
            name (str, optional): Name of the metric instance. Defaults to "r_squared".
        """
        super(R_squared, self).__init__(name=name, **kwargs)
        # Initialize internal variables necessary for computation
        self.residual = self.add_weight(name="residual", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulates statistics for the R-squared metric.

        Args:
            y_true (tf.Tensor): True values (ground truth).
            y_pred (tf.Tensor): Predicted values.
            sample_weight (tf.Tensor, optional): Optional sample weights.
        """
        # Calculate the residual sum of squares (deviation of predictions from truth)
        residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        # Calculate the total sum of squares (deviation from the mean)
        total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))

        # Update internal variables
        self.residual.assign_add(residual)
        self.total.assign_add(total)

    def result(self):
        """
        Compute and return the R-squared metric.

        Returns:
            tf.Tensor: The R-squared value.
        """
        # Calculate R-squared using accumulated statistics
        r_squared = 1 - tf.divide(self.residual, self.total)
        return r_squared

    def reset_states(self):
        """
        Resets all of the metric state variables.
        """
        # Reset the accumulated values for fresh computation
        self.residual.assign(0.0)
        self.total.assign(0.0)
