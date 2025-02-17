import os
import unittest

import numpy as np

import src


class TokenizerTestCase(unittest.TestCase):
    """ Tests for tokenizer functions """

    def setUp(self):
        """
        Tokenizers and sample texts and words
        """
        self.current_dir = os.path.dirname(__file__)

        self.char_mapper = src.features.vectorizer.Mapper(char_level=True)

        self.char_example = "José Raúl Capablanca$"

        self.char_map = {" ": 0, "$": 1, "a": 2,
                         "b": 3, "c": 4, "j": 5,
                         "l": 6, "n": 7, "o": 8,
                         "p": 9, "r": 10, "s": 11,
                         "é": 12, "ú": 13, "?": 14}
        self.n_map = {0: " ", 1: "$", 2: "a",
                      3: "b", 4: "c", 5: "j",
                      6: "l", 7: "n", 8: "o",
                      9: "p", 10: "r", 11: "s",
                      12: "é", 13: "ú", 14: "?"}

        self.seq_name = "Minamoto no Yoshitsune$"
        self.seq_name_corpus = ["Minamoto no Yoshitsune$"]

        self.seq_name_sequences = ["min", "ina", "nam", "amo", "mot",
                                   "oto", "to ", "o n", " no", "no ",
                                   "o y", " yo", "yos", "osh", "shi",
                                   "hit", "its", "tsu", "sun", "une"]
        self.seq_name_next = ["a", "m", "o", "t", "o",
                              " ", "n", "o", " ", "y",
                              "o", "s", "h", "i", "t",
                              "s", "u", "n", "e", "$"]

        self.ood_name = "Benkei"
        self.ood_name_sequences = ["?en", "en?", "n?e"]
        self.ood_name_next = ["?", "e", "i"]

        seq_example_vectors = os.path.join(self.current_dir, "test_files/features/vectorize_char.npy")
        with open(seq_example_vectors, 'rb') as f:
            self.seq_name_sequences_vector = np.load(f)
            self.seq_name_next_vector = np.load(f)

        self.char_sequenizer = src.features.vectorizer.SequenceVectorizer(corpus=self.seq_name_corpus,
                                                                             char_level=True)

    def test_tokenize_characters(self):
        example_clean, char_map_test, n_map_test = self.char_mapper.create_text_map(self.char_example)
        self.assertDictEqual(self.char_map, char_map_test)
        self.assertDictEqual(self.n_map, n_map_test)

    def test_sequenize_text(self):
        name_sequences_test, name_next_chars_test = self.char_sequenizer.sequenize(self.seq_name,
                                                                                   maxlen=3)
        self.assertListEqual(self.seq_name_sequences, name_sequences_test)
        self.assertListEqual(self.seq_name_next, name_next_chars_test)

        ood_sequences_test, ood_next_chars_test = self.char_sequenizer.sequenize(self.ood_name,
                                                                                 maxlen=3)
        self.assertListEqual(self.ood_name_sequences, ood_sequences_test)
        self.assertListEqual(self.ood_name_next, ood_next_chars_test)

    def test_vectorize_text(self):
        name_sequences_vector_test, name_next_vector_test = self.char_sequenizer.vectorize(self.seq_name,
                                                                                           maxlen=3)
        np.testing.assert_array_equal(self.seq_name_sequences_vector, name_sequences_vector_test)
        np.testing.assert_array_equal(self.seq_name_next_vector, name_next_vector_test)
