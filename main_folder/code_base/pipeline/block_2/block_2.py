import tensorflow as tf
from code_base.pipeline.block_1.dcl import DCL
from tensorflow.python.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    LeakyReLU,
    ReLU,
    SeparableConv2D,
)


class Block2(tf.keras.layers.Layer):
    def __init__(self):
      super().__init__()

    def call(self, inputs):
        x = DCL()(inputs)
        # This is the input to the dense block
        # First SeparableConv2D layer
        layer1 = SeparableConv2D(
            32,
            (3, 3),
            strides=(1, 1),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            use_bias=False,
        )(x)
        layer1 = BatchNormalization()(layer1)
        layer1 = LeakyReLU(alpha=0.2)(layer1)

        # Concatenate input with the output of the first layer
        # concat1 = Add()([x, layer1])

        # Second SeparableConv2D layer
        layer2 = SeparableConv2D(
            32,
            (3, 3),
            strides=(1, 1),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            use_bias=False,
        )(layer1)
        layer2 = BatchNormalization()(layer2)
        layer2 = LeakyReLU(alpha=0.2)(layer2)

        # Concatenate input, first layer output, and second layer output
        concat2 = Add()([layer1, layer2])

        # Third SeparableConv2D layer
        layer3 = SeparableConv2D(
            32,
            (3, 3),
            strides=(1, 1),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            use_bias=False,
        )(concat2)
        layer3 = BatchNormalization()(layer3)
        layer3 = LeakyReLU(alpha=0.2)(layer3)

        # Final concatenation of all layers' outputs
        output = Add()([concat2, layer3])

        return output
