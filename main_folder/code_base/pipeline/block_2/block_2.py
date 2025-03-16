import tensorflow as tf
# from code_base.pipeline.block_1.dcl import DCL
from tensorflow.keras.layers import (
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

    def build(self):
        self.conv2d = [SeparableConv2D(
            32,
            (3, 3),
            strides=(1, 1),
            padding="same",
            depthwise_initializer="he_normal",
            pointwise_initializer="he_normal",
            use_bias=False,
        ) for _ in range(3)]
        self.bn = [BatchNormalization() for _ in range(3)]
        self.leaky = [LeakyReLU(alpha=0.2) for _ in range(3)]
        self.add = [Add() for _ in range(2)]

    def call(self, inputs):
        x = inputs
        layer1 = self.conv2d[0](x)
        layer1 = self.bn[0](layer1)
        layer1 = self.leaky[0](layer1)
        layer2 = self.conv2d[1](layer1)
        layer2 = self.bn[1](layer2)
        layer2 = self.leaky[1](layer2)
        concat2 = self.add[0]([layer1, layer2])
        layer3 = self.conv2d[2](concat2)
        layer3 = self.bn[2](layer3)
        layer3 = self.leaky[2](layer3)
        output = self.add[1]([concat2, layer3])
        return output
