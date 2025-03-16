import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import (
    Add,
)
# from code_base.pipeline.block_2.block_2 import Block2
# from code_base.pipeline.block_2.eca import ECALayer2
# from code_base.pipeline.block_1.dcl import DCL

class Concat2(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self):
        self.add = Add()

    def call(self, *inputs):
        channel_attention_map1, output = inputs
        x112 = self.add([channel_attention_map1, output])
        return x112
