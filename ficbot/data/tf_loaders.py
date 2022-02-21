import tensorflow as tf
import numpy as np
import os

from ficbot.features.mapper import SequenceVectorizer


class ImageLoader(object):

    def __init__(self, *args, **kwargs):  # a mixin: a class that is DESIGNED to be used for multiple inheritance.
        super().__init__(*args, **kwargs)  # forwards all unused arguments

    @staticmethod
    def _get_image(path, target_size):

        image = tf.keras.preprocessing.image.load_img(path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)

        image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()
        return image_arr/225.


class NameLoader(object):

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def _get_sequences(self, name: str, *,
                       maxlen: int,
                       step: int = 1):
        vector_sequences, vector_next = self.vectorizer.vectorize_text(name,
                                                                       maxlen=maxlen,
                                                                       step=step)
        return vector_sequences, vector_next


class ImgNameGenerator(ImageLoader, NameLoader, tf.keras.utils.Sequence):

    def __init__(self, df, img_col, name_col, *,
                 img_folder: str,
                 batch_size: int = 1,
                 image_shape: tuple = (224, 224, 3),
                 maxlen: int = 3,
                 step: int = 1,
                 shuffle: bool = True):

        self.df = df.copy()
        self.img_col = img_col
        self.name_col = name_col
        self.img_folder = img_folder
        self.batch_size = batch_size
        self.img_shape = image_shape
        self.maxlen = maxlen
        self.step = step
        self.shuffle = shuffle

        self.vectorizer = SequenceVectorizer(df[name_col].tolist())
        super().__init__(self.vectorizer)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_data(self, batches):
        name_batch = batches[self.name_col]
        img_batch = batches[self.img_col]

        img_paths = [os.path.join(self.img_folder, img_name) for img_name in img_batch]

        X_img_batch = np.asarray([self._get_image(img_path, self.img_shape) for img_path in img_paths])
        X_seq_batch, y_batch = [], []
        for name in name_batch:
            X_seq, y = self._get_sequences(name, maxlen=self.maxlen)
            X_seq_batch.append(X_seq)
            y_batch.append(y)
        X_seq_batch, y_batch = np.asarray(X_seq_batch), np.asarray(y_batch)

        return tuple([X_img_batch, X_seq_batch]), y_batch

    def __getitem__(self, index):

        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y
