import unittest
import os
import base64

from fastapi.testclient import TestClient

from src.inference.main import app
from src.inference.config import settings

from test.config import current_dir


class ContainerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        settings.testing = True

    def test_img2name_container(self):

        client = TestClient(app)

        response = client.get("/health/")
        self.assertEqual(response.status_code, 200)
        
        image_path = os.path.join(current_dir, 'files/sample.jpg')

        # Load and encode test image
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode()
        
        payload = {
            "image": encoded_image,
            "diversity": 1.2,
            "min_name_length": 2
        }

        response = client.post("/generate/", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn("name", response.json())
        self.assertTrue(len(response.json()["name"].split()) >= 2)
        print(response.json()["name"])
