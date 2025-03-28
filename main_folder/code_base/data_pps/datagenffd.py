from code_base.utils.constants import img_cols, img_rows
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

class PreprocessData:
    def transform(DIR, batch_size: int, shuffle: bool, class_mode=None):
        datagen = ImageDataGenerator(rescale=1 / 255.0)

        generator = datagen.flow_from_directory(
            DIR,
            batch_size=batch_size,
            class_mode=class_mode, # make it categorical for train and validation dir
            target_size=(img_rows, img_cols),
            shuffle=shuffle,
        )
        return generator
