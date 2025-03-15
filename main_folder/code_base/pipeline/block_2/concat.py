import tensorflow as tf
from keras import layers
from code_base.pipeline.block_2.block_2 import Block2
from code_base.pipeline.block_2.eca import ECALayer
from code_base.pipeline.block_1.dcl import DCL


class Concat(tf.keras.layers.Layer):
    def __init__(self):
      super().__init__()

    def call(self, inputs):
        x_offset = DCL()(inputs)
        output = Block2()(x_offset)
        # Create an instance of the ECALayer
        eca_layer = ECALayer(k_size=3)

        # Apply ECA module to the input
        channel_attention_map1 = eca_layer(x_offset)
        channel_attention_map1.shape

        x112 = layers.add([channel_attention_map1, output])
        return x112
