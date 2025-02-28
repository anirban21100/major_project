from block1 import Block1
from eca import ECALayer
from keras import layers
import tensorflow as tf
from ..preblock.pre_block import PreBlock
from tensorflow.python.keras.layers import Input

class Concat(tf.keras.layers.Layer):
    def __init__(self):
      super().__init__()
    def call(self, inputs):
        x_offset = inputs
        output = Block1()(x_offset)
        eca_layer = ECALayer(k_size=3)

        # Apply ECA module to the input
        channel_attention_map1 = eca_layer(x_offset)
        channel_attention_map1.shape

        x111 = layers.add([channel_attention_map1, output])
        return x111
