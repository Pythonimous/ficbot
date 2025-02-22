import unittest
import os

import tensorflow as tf

from src.core.models.img2name.img2name import Img2Name
from src.core.inference import generate_name
from test.config import current_dir

class TfModelsTestCase(unittest.TestCase):

    def test_compile(self):
        simple_name_tf = Img2Name(maxlen=3, vocab_size=420)
        simple_name_tf.compile(loss='categorical_crossentropy', optimizer='adam')
        self.assertIsInstance(simple_name_tf, tf.keras.models.Model)
        self.assertEqual(simple_name_tf.get_layer("IMAGE_INPUT").output.shape, (None, 224, 224, 3))
        self.assertEqual(simple_name_tf.get_layer("NAME_INPUT").output.shape, (None, 3, 420))
        self.assertEqual(simple_name_tf.get_layer("PREDICTION").output.shape, (None, 420))


    def test_generate_img2name(self):

        image_path = os.path.join(current_dir, '../example/name/1.jpg')
        model_path = os.path.join(current_dir, '../src/core/models/img2name/img2name.keras')
        maps_path = os.path.join(current_dir, '../src/core/models/img2name/maps.pkl')

        generated_name = generate_name(image_path, model_path, maps_path, diversity=1.2)
        self.assertIsInstance(generated_name, str)
