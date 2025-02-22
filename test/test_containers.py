import unittest
import os

import json
import base64
from test.config import current_dir

from fastapi.testclient import TestClient

class ContainerTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["TESTING"] = "1"

    def test_img2name_container(self):
        from src.core.models.img2name.container import app

        client = TestClient(app)

        response = client.get("/health/")
        self.assertEqual(response.status_code, 200)
        
        image_path = os.path.join(current_dir, '../example/name/1.jpg')

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
