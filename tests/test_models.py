import unittest

import keras

from ficbot.character.name.models import SimpleModel


class TfModelsTestCase(unittest.TestCase):

    def test_compile(self):
        simple_name_tf = SimpleModel(maxlen=3, vocab_size=420)
        simple_name_tf.compile(loss='categorical_crossentropy', optimizer='adam')
        self.assertEqual(type(simple_name_tf), keras.engine.functional.Functional)
        self.assertEqual(simple_name_tf.get_layer("IMAGE_INPUT").output.shape.as_list(), [None, 224, 224, 3])
        self.assertEqual(simple_name_tf.get_layer("NAME_INPUT").output.shape.as_list(), [None, 3, 420])
        self.assertEqual(simple_name_tf.get_layer("PREDICTION").output.shape.as_list(), [None, 420])
