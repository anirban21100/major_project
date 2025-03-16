# from code_base.pipeline.block_1.concat import Concat
import tensorflow as tf
from tensorflow.keras.layers import (
    ELU,
    BatchNormalization,
    LeakyReLU,
    MaxPooling2D,
    SeparableConv2D,
)

class DCL1(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self):
        self.conv2d = SeparableConv2D(
            32,
            (3, 3),
            strides=(1, 1),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            activation=ELU(),
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
        x111 = inputs
        x = self.conv2d(x111)
        x = self.bn(x)
        x_offset = self.leaky(x)
        x_offset = self.maxpool(x_offset)
        return x_offset
