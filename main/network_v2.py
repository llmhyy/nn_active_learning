import tensorflow as tf


class Network:
    def __init__(self, learning_rate=0.001, shape=None):
        self.learning_rate = learning_rate
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=shape),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        self.model.summary()

