import tensorflow as tf
from keras import layers
from code_base.pipeline.block_3.block_3 import Block3
from code_base.pipeline.block_3.eca import ECALayer
from code_base.pipeline.block_2.dcl import DCL


class Concat(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x_offset = DCL()(inputs)
        output = Block3()(inputs)
        eca_layer = ECALayer(k_size=3)
        channel_attention_map1 = eca_layer(x_offset)
        x113 = layers.add([channel_attention_map1, output])
        return x113
