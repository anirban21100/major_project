import tensorflow as tf
from keras import layers
from code_base.pipeline.block_1.block_1 import Block1
from code_base.pipeline.block_1.eca import ECALayer


class Concat(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x_offset = inputs
        output = Block1()(x_offset)
        eca_layer = ECALayer(k_size=3)
        channel_attention_map1 = eca_layer(x_offset)
        x111 = layers.add([channel_attention_map1, output])
        return x111
