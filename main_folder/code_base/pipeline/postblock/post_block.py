from block_4.dcl import DCL
from utils.constants import input_shape
import tensorflow as tf
from tensorflow.python.keras.layers import GlobalMaxPooling2D, Dense, Input
from tensorflow.python.keras.models import Model

class PostBlock(tf.keras.layers.Layer):
    def __init__(self):
      super().__init__()

    def call(self, inputs):
        x_offset = DCL()(inputs)
        x15 = GlobalMaxPooling2D()(x_offset)
        img_input = Input(input_shape)
        x = tf.keras.layers.Dropout(0.3)(x15)
        output=Dense(6, activation='softmax')(x)
        model = Model(img_input, output)
        return model