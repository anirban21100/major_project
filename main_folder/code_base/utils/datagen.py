import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from code_base.utils.constants import img_rows as ROWS
from code_base.utils.constants import img_cols as COLS

class DataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size, target_size=(ROWS, COLS), shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        X, y = self.__data_generation(batch_images, batch_labels)
        return (X, y), y

    def __data_generation(self, batch_images, batch_labels):
        X = np.array([img_to_array(load_img(img_path, target_size=self.target_size)) / 255.0 for img_path in batch_images])
        y = np.array(batch_labels)
        # y = tf.keras.utils.to_categorical(y, num_classes=15)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
