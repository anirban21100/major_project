from code_base.pipeline.block_4.concat import Concat
import tensorflow as tf
from tensorflow.python.keras.layers import (
    ELU,
    BatchNormalization,
    LeakyReLU,
    MaxPooling2D,
    SeparableConv2D,
)


class DCL(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x114 = Concat()(inputs)
        x = SeparableConv2D(
            256,
            (3, 3),
            strides=(1, 1),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            activation=LeakyReLU(alpha=0.2),
            use_bias=False,
        )(x114)
        x = BatchNormalization()(x)
        x_offset = LeakyReLU(alpha=0.2)(x)
        return x_offset
