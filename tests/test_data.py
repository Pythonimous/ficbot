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
        descriptions_beautified = [ficbot.data.download.beautify_bio(bio) for bio in self.descriptions_dirty]
        self.assertEqual(descriptions_beautified, self.descriptions_clean, 'Bios not beautified as expected.')


class PreprocessTestCase(unittest.TestCase):
    """ Tests for different preprocessing functions """

    def test_replace_text_numbers(self):

        self.assertEqual(ficbot.data.preprocessing.replace_text_numbers("Mayu Watanabe CG-3"),
                         "Mayu Watanabe CG-three")
        self.assertEqual(ficbot.data.preprocessing.replace_text_numbers("Pour Lui 13-sei"),
                         "Pour Lui thirteen-sei")
        self.assertEqual(ficbot.data.preprocessing.replace_text_numbers("Asuka Kuramochi the 9th"),
                         "Asuka Kuramochi the ninth")
        self.assertEqual(ficbot.data.preprocessing.replace_text_numbers("Yui Yokoyama the 7.5th"),
                         "Yui Yokoyama the seven point fifth")
        self.assertEqual(ficbot.data.preprocessing.replace_text_numbers("02"),
                         "zero two")
        self.assertEqual(ficbot.data.preprocessing.replace_text_numbers(".01"),
                         ".zero one")
        self.assertEqual(ficbot.data.preprocessing.replace_text_numbers("The 03 is a lie"),
                         "The zero three is a lie")

    def test_clean_text_from_symbols(self):

        self.assertEqual(ficbot.data.preprocessing.clear_text("Ángela Salas Larrazábal"),
                         "Angela Salas Larrazabal"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Simo Häyhä"),
                         "Simo Haeyhae"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Christine Waldegård"),
                         "Christine Waldegaard"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Selim Vergès"),
                         "Selim Verges"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Padmé Amidala"),
                         "Padme Amidala"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Pierre Tempête de Neige"),
                         "Pierre Tempete de Neige"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Chloë Maxwell"),
                         "Chloe Maxwell"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Bernardo Dión"),
                         "Bernardo Dion"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Gérôme Hongou"),
                         "Geroome Hongou"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Arad Mölders"),
                         "Arad Moelders"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Tor Nørretranders"),
                         "Tor Noerretranders"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Jürgen von Klügel"),
                         "Juergen von Kluegel"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Œlaf"),
                         "Oelaf"),
        self.assertEqual(ficbot.data.preprocessing.clear_text("Daša Urban"),
                         "Dasha Urban")
        self.assertEqual(ficbot.data.preprocessing.clear_text("02,';'1"),
                         "Zero two one")
        self.assertEqual(ficbot.data.preprocessing.clear_text("Åll your 1.2 bases are. SO bel']ong to-us 13"),
                         "Aall your one point two bases are. SO bel ong to-us thirteen")


class TfLoadersTestCase(unittest.TestCase):
    """ Tests for tensorflow loaders """

    def setUp(self):
        self.current_dir = os.path.dirname(__file__)
        self.df_path = os.path.join(self.current_dir, "test_files/data/tf_loaders/img_name.csv")
        self.df = pd.read_csv(self.df_path)
        self.img_folder = os.path.join(self.current_dir, "test_files/data/tf_loaders/images")
        self.image_shape = (224, 224, 3)
        self.maxlen = 3
        self.batch_size = 5
        self.tf_img_name_loader = ficbot.data.loaders.tf_loaders.ImgNameLoader(self.df, "image", "eng_name",
                                                                               batch_size=self.batch_size,
                                                                               img_folder=self.img_folder,
                                                                               image_shape=self.image_shape,
                                                                               maxlen=self.maxlen)

    def test_img_name_loader(self):
        n_sequences = 0
        for i in range(self.batch_size):
            n_sequences += len(self.df.eng_name[i]) + 1
        vocab_size = self.tf_img_name_loader.vectorizer.get_vocab_size()
        X, y = self.tf_img_name_loader.__getitem__(0)
        X_img_batch, X_seq_batch = X
        expected_img_batch_shape = (n_sequences, ) + self.image_shape
        self.assertEqual(X_img_batch.shape, expected_img_batch_shape)
        self.assertEqual(X_seq_batch.shape, (n_sequences, self.maxlen, vocab_size))
        self.assertEqual(y.shape, (n_sequences, vocab_size))
