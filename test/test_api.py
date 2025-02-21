import os
import unittest
import time

from fastapi.testclient import TestClient

from src.main import app
from src.api.utils import UPLOAD_DIR, clean_old_images

from test.config import current_dir

client = TestClient(app)  # Test client for simulating API requests

class TestAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["TESTING"] = "1"

    def test_root_endpoint(self):
        """Test the root endpoint (health check)."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


    def test_upload_image_endpoint(self):
        """Test the image upload endpoint with a valid image."""

        example_path = UPLOAD_DIR / "example.jpg"
        self.assertTrue(example_path.exists())

        img1_path = os.path.join(current_dir, "files/sample.jpg")

        with open(img1_path, "rb") as image:
            files = {"file": ("sample.jpg", image, "image/jpeg")}
            response = client.post("/upload_image/", files=files)

        self.assertEqual(response.status_code, 200)
        self.assertIn("imgUrl", response.json())

        self.assertTrue(example_path.exists())  # example.jpg is still there
        img1_path = UPLOAD_DIR / response.json()["imgUrl"].split("/")[-1]
        self.assertTrue(img1_path.exists())

        time.sleep(3)  # for testing image timeout is 2 seconds

        img2_path = os.path.join(current_dir, "files/sample.jpg")
        with open(img2_path, "rb") as image:
            files = {"file": ("sample.jpg", image, "image/jpeg")}
            response = client.post("/upload_image/", files=files)

        self.assertEqual(response.status_code, 200)
        self.assertIn("imgUrl", response.json())

        self.assertTrue(example_path.exists())  # example.jpg is still there
        img2_path = UPLOAD_DIR / response.json()["imgUrl"].split("/")[-1]
        self.assertTrue(img2_path.exists())  # new file is now uploaded
        self.assertFalse(img1_path.exists())  # old file is now removed since it is older than 2 seconds


    def test_upload_image_wrong_extension(self):
        """Test uploading a file with an invalid extension."""

        txt_path = os.path.join(current_dir, "files/sample.txt")

        with open(txt_path, "rb") as text_file:
            files = {"file": ("sample.txt", text_file, "text/plain")}
            response = client.post("/upload_image/", files=files)

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "Wrong extension: only .jpg, .png, .gif files are allowed")

    
    def test_upload_invalid_image(self):
        """Test uploading an invalid image."""

        img_path = os.path.join(current_dir, "files/invalid.jpg")

        with open(img_path, "rb") as image:
            files = {"file": ("invalid.jpg", image, "image/jpeg")}
            response = client.post("/upload_image/", files=files)

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "Broken file: only valid .jpg, .png, .gif files are allowed. Please check your image and try again.")


    def test_upload_image_too_large(self):
        """Test uploading a file that is too large."""

        img_path = os.path.join(current_dir, "files/large.jpg")

        with open(img_path, "rb") as image:
            files = {"file": ("large.jpg", image, "image/jpeg")}
            response = client.post("/upload_image/", files=files)

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "File is too large. Only .jpg, .png, .gif up to 2MB are allowed.")

    
    def test_generate_character_name(self):
        """Test the name generation API with valid parameters."""
        
        img_path = os.path.join("http://127.0.0.1:8000/static/images/example.jpg")
        payload = {
            "imageSrc": img_path,
            "diversity": 1.2,
            "min_name_length": 4
        }
        response = client.post("/name/", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn("name", response.json())
        self.assertIsInstance(response.json()["name"], str)


    def test_generate_character_name_invalid_image(self):
        """Test the name generation API with a non-existing image."""
        img_path = os.path.join(current_dir, "files/non_existent.jpg")
        payload = {
            "imageSrc": img_path,
            "diversity": 1.2,
            "min_name_length": 4
        }
        response = client.post("/name/", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Image file not found")
    