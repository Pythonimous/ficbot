import os
import json
import numpy as np
import tensorflow as tf
import base64
import pickle
from io import BytesIO

# Load the model and mappings (only once per Lambda cold start)
MODEL_PATH = "/tmp/img2name.keras"
MAPS_PATH = "/tmp/maps.pkl"

if os.environ.get("TESTING"):
    MODEL_PATH = "src/core/models/img2name/img2name.keras"
    MAPS_PATH = "src/core/models/img2name/maps.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(MAPS_PATH, "rb") as mp:
    char_idx, idx_char = pickle.load(mp)
    

def sample(preds, temperature=1.0):
    """ Samples an index from a probability array using temperature scaling. """
    preds = np.asarray(preds, dtype=np.float64)
    preds = np.log(preds + 1e-10) / temperature  # Avoid log(0)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))


def load_image_for_model(image_bytes):
    """ Converts base64 image bytes into a processed feature vector. """
    target_size = model.get_layer("IMAGE_INPUT").output.shape[1:]  # Model expected shape

    # Load image from bytes
    image = tf.keras.preprocessing.image.load_img(BytesIO(image_bytes))

    # Convert to array, resize, and normalize
    image_arr = tf.keras.preprocessing.image.img_to_array(image)
    image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()
    image_arr = tf.keras.applications.mobilenet.preprocess_input(image_arr)

    return np.expand_dims(image_arr, axis=0)  # Add batch dimension


def generate_name(image_bytes, min_name_length=2, diversity=1.2,
                  start_token="@", end_token="$", ood_token="?"):
    """ Generates a name from an image using the trained model. """

    image_features = load_image_for_model(image_bytes)
    maxlen = model.get_layer("NAME_INPUT").output.shape[1]

    generated = ""
    name = start_token * maxlen

    while not (generated.endswith(start_token) or generated.endswith(end_token)):
        x_pred_text = np.zeros((1, maxlen, len(idx_char)))

        for t, char in enumerate(name):
            x_pred_text[0, t, char_idx[char]] = 1.0

        preds = model.predict([image_features, x_pred_text], verbose=0)[0]

        next_char = ood_token
        while next_char == ood_token:  # Keep sampling until a valid char is found
            next_index = sample(preds, diversity)
            next_char = idx_char[next_index]

        if next_char == end_token and generated.count(' ') < min_name_length - 1:
            next_char = " "  # Ensure minimum length

        name = name[1:] + next_char
        generated += next_char

    # Cleanup
    generated = generated.rstrip(start_token + end_token)
    generated = " ".join(word.capitalize() for word in generated.split())

    return generated


def lambda_handler(event, context):
    """ AWS Lambda entry point for handling inference requests. """
    try:
        # Parse request body
        body = json.loads(event["body"])
        
        # Decode base64 image
        image_bytes = base64.b64decode(body["image"])

        # Extract parameters
        diversity = float(body["diversity"])
        min_name_length = int(body["min_name_length"])

        # Generate name
        name = generate_name(image_bytes, min_name_length=min_name_length, diversity=diversity)

        return {
            "statusCode": 200,
            "body": json.dumps({"name": name})
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }