import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import (
    Add,
)
# from code_base.pipeline.block_1.block_1 import Block1
# from code_base.pipeline.block_1.eca import ECALayer1

class Concat1(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self):
        self.add = Add()

    def call(self, *inputs):
        channel_attention_map1, output = inputs
        x111 = self.add([channel_attention_map1, output])
        return x111
