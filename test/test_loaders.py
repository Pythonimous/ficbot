import unittest
import os

import pandas as pd

from src.core.loaders import ImgNameLoader


class TfLoadersTestCase(unittest.TestCase):
    """ Tests for tensorflow loaders """

    def setUp(self):
        self.current_dir = os.path.dirname(__file__)
        self.df_path = os.path.join(self.current_dir, "files/data/loaders/img_name.csv")
        self.df = pd.read_csv(self.df_path)
        self.img_dir = os.path.join(self.current_dir, "files/data/loaders/images")
        self.image_shape = (224, 224, 3)
        self.maxlen = 3
        self.batch_size = 5
        self.img_name_loader = ImgNameLoader(self.df, "image", "eng_name",
                                                batch_size=self.batch_size,
                                                img_dir=self.img_dir,
                                                image_shape=self.image_shape,
                                                maxlen=self.maxlen)

    def test_img_name_loader(self):
        n_sequences = 0
        for i in range(self.batch_size):
            n_sequences += len(self.df.eng_name[i]) + 1
        vocab_size = self.img_name_loader.vectorizer.get_vocab_size()
        X, y = self.img_name_loader.__getitem__(0)
        X_img_batch, X_seq_batch = X
        expected_img_batch_shape = (n_sequences, ) + self.image_shape
        self.assertEqual(X_img_batch.shape, expected_img_batch_shape)
        self.assertEqual(X_seq_batch.shape, (n_sequences, self.maxlen, vocab_size))
        self.assertEqual(y.shape, (n_sequences, vocab_size))
