import unittest
import os
import pandas as pd

from .context import ficbot


class DownloadTestCase(unittest.TestCase):
    """ Tests for link and character page download functions """
    def setUp(self):
        """
        Character descriptions which should be beautified
        """
        self.descriptions_dirty = [
            "Here's a description.\n(Source: Somewebsite)",
            "\n\nHere's another description.\n\nHere's one more.",
            "       Here's the last description.\n\n           ",
            "No biography written."
        ]
        self.descriptions_clean = [
            "Here's a description.",
            "Here's another description.\nHere's one more.",
            "Here's the last description.",
            ""
        ]

    def test_beautify(self):
        descriptions_beautified = [ficbot.data.download_data.beautify_bio(bio) for bio in self.descriptions_dirty]
        self.assertEqual(descriptions_beautified, self.descriptions_clean, 'Bios not beautified as expected.')


class TfLoadersTestCase(unittest.TestCase):
    """ Tests for tensorflow loaders """

    def setUp(self):
        self.current_dir = os.path.dirname(__file__)
        self.df_path = os.path.join(self.current_dir, "test_files/data/tf_loaders/img_name.csv")
        self.df = pd.read_csv(self.df_path)
        self.img_folder = os.path.join(self.current_dir, "test_files/data/tf_loaders/images")
        self.image_shape = (224, 224, 3)
        self.maxlen = 3
        self.batch_size = 1
        self.tf_img_name_loader = ficbot.data.tf_loaders.ImgNameGenerator(self.df, "image", "eng_name",
                                                                          batch_size=self.batch_size,
                                                                          img_folder=self.img_folder,
                                                                          image_shape=self.image_shape,
                                                                          maxlen=self.maxlen)

    def test_img_name_loader(self):
        n_sequences = len(self.df.eng_name[0]) + 1
        vocab_size = self.tf_img_name_loader.vectorizer.get_vocab_size()
        X, y = self.tf_img_name_loader.__getitem__(0)
        X_img_batch, X_seq_batch = X
        expected_img_batch_shape = (self.batch_size, ) + self.image_shape
        self.assertEqual(X_img_batch.shape, expected_img_batch_shape)
        self.assertEqual(X_seq_batch.shape, (self.batch_size, n_sequences, self.maxlen, vocab_size))
        self.assertEqual(y.shape, (self.batch_size, n_sequences, vocab_size))
