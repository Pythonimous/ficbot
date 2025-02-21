import unittest
import os

import json
import base64
from test.config import current_dir

class TfLambdaTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["TESTING"] = "1"

    def test_img2name_lambda(self):
        from src.core.models.img2name.lambda_function import lambda_handler
        
        image_path = os.path.join(current_dir, '../example/name/1.jpg')

        # Load and encode test image
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode()
        
        mock_event = {
            "body": json.dumps({
                "image": encoded_image,
                "diversity": 1.2,
                "min_name_length": 2
            })
        }

        # Call the Lambda function locally
        response = lambda_handler(mock_event, None)
        response_body = json.loads(response["body"])

        self.assertEqual(response["statusCode"], 200)
        self.assertTrue("name" in response_body)
        self.assertTrue(len(response_body["name"].split()) >= 2)
        print(response_body["name"])
