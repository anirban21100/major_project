# from code_base.pipeline.block_4.dcl import DCL
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Input, Dropout
# from tensorflow.keras.models import Model


class PostBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        x_offset = inputs
        x15 = GlobalMaxPooling2D()(x_offset)
        x = Dropout(0.3)(x15)
        output = Dense(6, activation="softmax")(x)
        return output
