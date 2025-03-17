# from code_base.pipeline.block_4.dcl import DCL
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Input, Dropout
# from tensorflow.keras.models import Model

class PostBlock(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def build(self):
        self.globalpool = GlobalMaxPooling2D()
        self.dropout = Dropout(0.3)
        self.dense = Dense(self.num_classes, activation="softmax")

    def call(self, inputs):
        x_offset = inputs
        x15 = self.globalpool(x_offset)
        x = self.dropout(x15)
        output =self.dense(x)
        return output
