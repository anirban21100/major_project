import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    LeakyReLU,
    ReLU,
    SeparableConv2D,
    Input,
)

class Block1(tf.keras.layers.Layer):
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
        self.conv2d1 = self.conv2d
        self.conv2d2 = self.conv2d
        self.conv2d3 = self.conv2d
        self.bn = BatchNormalization()
        self.leaky = LeakyReLU(alpha=0.2)
        self.add = Add()

    def call(self, inputs):
        x = inputs
        layer1 = self.conv2d1(x)
        layer1 = self.bn(layer1)
        layer1 = self.leaky(layer1)
        layer2 = self.conv2d2(layer1)
        layer2 = self.bn(layer2)
        layer2 = self.leaky(layer2)
        concat2 = self.add([layer1, layer2])
        layer3 = self.conv2d3(concat2)
        layer3 = self.bn(layer3)
        layer3 = self.leaky(layer3)
        output = self.add([concat2, layer3])
        return output
