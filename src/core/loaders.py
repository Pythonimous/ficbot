import pandas as pd
import tensorflow as tf
import numpy as np
import os

from src.core.utils import get_image
from src.features.vectorizer import SequenceVectorizer


class ImageLoader(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class NameLoader(object):
    """Vectorizes name using provided vectorizer."""
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def _get_sequences(self, name: str, *,
                       maxlen: int,
                       step: int = 1):
        vector_sequences, vector_next = self.vectorizer.vectorize(name,
                                                                  maxlen=maxlen,
                                                                  step=step)
        return vector_sequences, vector_next


class ImgNameLoader(ImageLoader, NameLoader, tf.keras.utils.Sequence):
    """Image + Name loader.

    Attributes:
        df (Pandas dataframe): dataframe with a name column
        img_col (str): Label of a column with image names
        name_col (str): Label of a column with character names
        batch-size (int): Batch size
        img_dir (str): Path to a directory with images
        img_shape (tuple): Desired image shape for the pretrained algorithm
        transfer_net (str): What pretrained network to use for transfer learning.
            Currently available: [mobilenet]
        vectorizer (features.vectorizer.SequenceVectorizer object):
            a preinitializer name vectorizer using existing maps (optional)
        start_token (str): a token to denote a beginning of a name
        end_token (str): a token to denote an end of a name
        maxlen (str): what length the sequences should be
        step (str): what step the sequenizer should take
        shuffle (bool): whether to shuffle the data after each epoch
    """

    def __init__(self, df, img_col, name_col, *,
                 batch_size: int = 1,

                 img_dir: str,
                 image_shape: tuple = (224, 224, 3),
                 transfer_net: str = "mobilenet",

                 vectorizer=None,
                 start_token: str = "@",
                 end_token: str = "$",
                 maxlen: int = 3,
                 step: int = 1,

                 shuffle: bool = True):

        self.df = df.copy()
        self.img_col = img_col
        self.name_col = name_col
        self.df[name_col] = self.df[self.name_col].map(lambda x: start_token * maxlen + x + end_token)

        self.batch_size = batch_size

        self.img_dir = img_dir
        self.img_shape = image_shape
        self.transfer_net = transfer_net

        self.maxlen = maxlen
        self.step = step
        self.shuffle = shuffle
        self.n = len(self.df)

        if vectorizer is None:
            self.vectorizer = SequenceVectorizer(corpus=self.df[name_col].tolist(), ood_token="?")
        else:
            self.vectorizer = vectorizer
        super().__init__(self.vectorizer)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_data(self, batches):
        name_batch = batches[self.name_col]
        img_batch = batches[self.img_col]
        img_paths = [os.path.join(self.img_dir, img_name) for img_name in img_batch]

        X_img_batch, X_seq_batch, y_batch = [], [], []

        for idx in range(len(name_batch)):
            name = name_batch.iloc[idx]
            image = get_image(img_paths[idx], self.img_shape, self.transfer_net)
            X_seq, y = self._get_sequences(name, maxlen=self.maxlen)
            for sequence_idx in range(len(X_seq)):
                X_img_batch.append(image)
                X_seq_batch.append(X_seq[sequence_idx])
                y_batch.append(y[sequence_idx])

        X_img_batch, X_seq_batch, y_batch = np.asarray(X_img_batch), np.asarray(X_seq_batch), np.asarray(y_batch)

        return tuple([X_img_batch, X_seq_batch]), y_batch

    def get_maxlen(self):
        return self.maxlen

    def __getitem__(self, index):

        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size


def create_loader(data_path, *, load_for, **kwargs):
    """ Assistant function to create a loader from given data.
    Args:
        data_path (string): Path to the dataframe
        loader (str): Name of the desired loader
        **kwargs: required and optional arguments for the desired loader
                  (check desired loader __init__ for details)
    """
    dataframe = pd.read_csv(data_path)
    loaders = {"simple_img_name"}
    assert load_for in loaders, f"No such loader.\nAvailable loaders: {', '.join(list(loaders))}."
    if load_for == "simple_img_name":
        return ImgNameLoader(dataframe, kwargs["img_col"], kwargs["name_col"],
                             img_dir=kwargs["img_dir"], batch_size=kwargs.get("batch_size", 1),
                             image_shape=kwargs.get("img_shape", (224, 224, 3)),
                             transfer_net=kwargs.get("transfer_net", "mobilenet"),
                             vectorizer=kwargs.get("vectorizer", None),
                             start_token=kwargs.get("start_token", "@"), end_token=kwargs.get("end_token", "$"),
                             maxlen=kwargs.get("maxlen", 3), step=kwargs.get("step", 1),
                             shuffle=kwargs.get("shuffle", True))

