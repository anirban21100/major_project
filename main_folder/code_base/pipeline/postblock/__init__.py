# x15=GlobalAveragePooling2D()(x11)
# x15
from tensorflow.keras.layers import GlobalMaxPooling2D

x15 = GlobalMaxPooling2D()(x_offset)

# x15=GlobalAveragePooling2D()(x_offset)
x15

# Dropout layer
# x=Dropout(0.3)(x15)
x = tf.keras.layers.Dropout(0.3)(x15)

from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    concatenate,
)
from keras.models import Model

# x = keras.layers.GlobalAveragePooling2D()(x33)
output = Dense(6, activation="softmax")(x)
output

model = Model(img_input, output)

model.summary()
