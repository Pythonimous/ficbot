import unittest

import tensorflow as tf

from src.training.models.img2name import Img2Name

class TestModels(unittest.TestCase):

    def test_compile(self):
        simple_name_tf = Img2Name(maxlen=3, vocab_size=420)
        simple_name_tf.compile(loss='categorical_crossentropy', optimizer='adam')
        self.assertIsInstance(simple_name_tf, tf.keras.models.Model)
        self.assertEqual(simple_name_tf.get_layer("IMAGE_INPUT").output.shape, (None, 224, 224, 3))
        self.assertEqual(simple_name_tf.get_layer("NAME_INPUT").output.shape, (None, 3, 420))
        self.assertEqual(simple_name_tf.get_layer("PREDICTION").output.shape, (None, 420))
