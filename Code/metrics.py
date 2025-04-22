"""Defines a custom MSE metric for evaluating the CycleGAN."""

import tensorflow as tf


class MSE_Metric(tf.keras.metrics.Metric):
    def __init__(self, max_val=1.0, name='MSE', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse = self.add_weight(name='MSE',
                                   initializer='zeros',
                                   )
        self.counter = self.add_weight(name='count',
                                       initializer='zeros')
        self.max_val = max_val

    def update_state(self, y_true, y_pred):       
        self.mse.assign_add(tf.math.reduce_sum(
            tf.square(tf.math.subtract(y_true, y_pred))))
        for _ in y_true:
            self.counter.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.mse,
                                     self.counter,
                                     )
    def reset_state(self):
        self.mse.assign(0.0)
        self.counter.assign(0.0)
