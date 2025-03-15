import tensorflow as tf
from keras import layers
from code_base.pipeline.block_4.block_4 import Block4
from code_base.pipeline.block_4.eca import ECALayer
from code_base.pipeline.block_3.dcl import DCL

class Concat(tf.keras.layers.Layer):
    def __init__(self):
      super().__init__()

    def call(self, inputs):
        x_offset = DCL()(inputs)
        output = Block4()(inputs)
        eca_layer = ECALayer(k_size=3)

        # Apply ECA module to the input
        channel_attention_map1 = eca_layer(x_offset)
        channel_attention_map1.shape

        x114 = layers.add([channel_attention_map1, output])
        return x114