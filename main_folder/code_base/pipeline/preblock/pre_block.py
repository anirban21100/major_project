import tensorflow as tf

# from tensorflow import keras
from tensorflow.keras.layers import (
    ELU,
    BatchNormalization,
    Input,
    LeakyReLU,
    PReLU,
    SeparableConv2D,
)
from tensorflow.python.keras.regularizers import l2

# from utils.constants import input_shape


class PreBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, img_input):
        x = SeparableConv2D(
            16,
            (3, 3),
            strides=(1, 1),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            use_bias=False,
        )(img_input)
        x = BatchNormalization()(x)
        x_offset = LeakyReLU(alpha=0.2)(x)

        return x_offset
