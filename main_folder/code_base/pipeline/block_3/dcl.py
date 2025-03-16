# from code_base.pipeline.block_3.concat import Concat
import tensorflow as tf
from tensorflow.keras.layers import (
    ELU,
    BatchNormalization,
    LeakyReLU,
    MaxPooling2D,
    SeparableConv2D,
)

class DCL3(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self):
        self.conv2d = SeparableConv2D(
            128,
            (3, 3),
            strides=(1, 1),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            activation=LeakyReLU(alpha=0.2),
            use_bias=False,
        )
        self.bn = BatchNormalization()
        self.leaky = LeakyReLU(alpha=0.2)
        self.maxpool = MaxPooling2D(
            (3, 3),
            strides=(2, 2),
            padding="same"
        )

    def call(self, inputs):
        x113 = inputs
        x = self.conv2d(x113)
        x = self.bn(x)
        x_offset = self.leaky(x)
        x_offset = self.maxpool(x_offset)
        return x_offset
