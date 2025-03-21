import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Input, Dropout, Softmax
from code_base.utils.ArcFace import ArcMarginProduct


class PostBlock(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def build(self):
        self.globalpool = GlobalMaxPooling2D()
        self.dropout = Dropout(0.3)
        self.label = Input(shape=())
        self.dense = Dense(self.num_classes, activation="softmax")
        self.margin = ArcMarginProduct(
            n_classes=self.num_classes, s=30, m=0.5, dtype="float32"
        )
        self.softmax = Softmax(dtype="float32")

    def call(self, inputs):
        x_offset = inputs
        x15 = self.globalpool(x_offset)
        x = self.dropout(x15)
        output = x
        # x = self.margin([x, self.label])
        # output = self.softmax(x)
        # self.dense(x)
        return output
