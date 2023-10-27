import tensorflow as tf

class R_squared(tf.keras.metrics.Metric):
    def __init__(self, name='r_squared', **kwargs):
        super(R_squared, self).__init__(name=name, **kwargs)
        self.squared_sum = self.add_weight(name='squared_sum', initializer='zeros')
        self.sum = self.add_weight(name='sum', initializer='zeros')
        self.residual = self.add_weight(name='residual', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate the residual sum of squares
        residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
        # Calculate the total sum of squares
        total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
        
        self.residual.assign_add(residual)
        self.total.assign_add(total)

    def result(self):
        # Calculate R-squared
        r_squared = 1 - tf.divide(self.residual, self.total)
        return r_squared

    def reset_states(self):
        # Reset the state of the metric
        self.residual.assign(0.0)
        self.total.assign(0.0)
