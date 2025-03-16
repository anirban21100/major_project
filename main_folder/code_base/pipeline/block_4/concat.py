import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import (
    Add,
)
# from code_base.pipeline.block_4.block_4 import Block4
# from code_base.pipeline.block_4.eca import ECALayer4
# from code_base.pipeline.block_3.dcl import DCL

class Concat4(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self):
        self.add = Add()

    def call(self, *inputs):
        channel_attention_map1, output = inputs
        x114 = self.add([channel_attention_map1, output])
        return x114
