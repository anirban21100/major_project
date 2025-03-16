import tensorflow as tf
from tensorflow.keras.layers import  (
    Conv1D,
    GlobalAveragePooling2D,
    Multiply,
    Reshape,
)

class ECALayer4(tf.keras.layers.Layer):
    def __init__(self, k_size=3, **kwargs):
        super().__init__()
        self.k_size = k_size

    def build(self):
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.conv1d = tf.keras.layers.Conv1D(
            filters=1,
            kernel_size=self.k_size,
            padding="same",
            use_bias=False
        )
        self.sigmoid = tf.keras.layers.Activation("sigmoid")

    def call(self, inputs):
        y = self.avg_pool(inputs)  # Shape: (batch_size, 1, 1, channels)
        y = tf.squeeze(y, axis=[1, 2])  # Shape: (batch_size, channels)
        y = self.conv1d(tf.expand_dims(y, axis=-1))  # Shape: (batch_size, channels, 1)
        y = tf.squeeze(y, axis=-1)  # Shape: (batch_size, channels)
        y = self.sigmoid(y)  # Shape: (batch_size, channels)
        y = tf.reshape(
            y, (-1, 1, 1, y.shape[-1])
        )  # Shape: (batch_size, 1, 1, channels)
        return inputs * y
