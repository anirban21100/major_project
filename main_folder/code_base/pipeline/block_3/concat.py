import tensorflow as tf
from keras import layers
from code_base.pipeline.block_3.block_3 import Block3
from code_base.pipeline.block_3.eca import ECALayer3
from code_base.pipeline.block_2.dcl import DCL


class Concat3(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, *inputs):
        channel_attention_map1, output = inputs
        x113 = layers.add([channel_attention_map1, output])
        return x113
