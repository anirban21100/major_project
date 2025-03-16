import tensorflow as tf
from tensorflow.keras.layers import (
    ELU,
    BatchNormalization,
    Input,
    LeakyReLU,
    PReLU,
    SeparableConv2D,
)
from tensorflow.keras.regularizers import l2

class PreBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self):
        self.conv2d = SeparableConv2D(
            16,
            (3, 3),
            strides=(1, 1),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            use_bias=False,
        )
        self.bn = BatchNormalization()
        self.leaky = LeakyReLU(alpha=0.2)

    def call(self, img_input):
        x = self.conv2d(img_input)
        x = self.bn(x)
        x_offset = self.leaky(x)
        return x_offset

    def compute_output_shape(self, input_shape):
        return input_shape
