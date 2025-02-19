import os
import unittest

from fastapi.testclient import TestClient

from src.api.main import app

from test.config import current_dir

client = TestClient(app)  # Test client for simulating API requests

class TestAPI(unittest.TestCase):


    def test_root_endpoint(self):
        """Test the root endpoint (health check)."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Welcome to the Ficbot API!"})


    def test_upload_image_endpoint(self):
        """Test the image upload endpoint with a valid image."""

        img_path = os.path.join(current_dir, "files/sample.jpg")

        with open(img_path, "rb") as image:
            files = {"file": ("sample.jpg", image, "image/jpeg")}
            response = client.post("/generate/upload_image/", files=files)

        self.assertEqual(response.status_code, 200)
        self.assertIn("imgUrl", response.json())


    def test_upload_image_wrong_extension(self):
        """Test uploading a file with an invalid extension."""

        txt_path = os.path.join(current_dir, "files/sample.txt")

        with open(txt_path, "rb") as text_file:
            files = {"file": ("sample.txt", text_file, "text/plain")}
            response = client.post("/generate/upload_image/", files=files)

        self.assertEqual(response.status_code, 415)
        self.assertEqual(response.json()["detail"], "Wrong extension: only .jpg, .png, .gif files are allowed")

    
    def test_generate_character_name(self):
        """Test the name generation API with valid parameters."""
        
        img_path = os.path.join("http://127.0.0.1:8000/static/images/example.jpg")
        payload = {
            "imageSrc": img_path,
            "diversity": 1.2,
            "min_name_length": 4
        }
        response = client.post("/generate/name/", json=payload)

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
        response = client.post("/generate/name/", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Image file not found")