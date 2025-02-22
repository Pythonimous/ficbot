import os
import numpy as np
import tensorflow as tf
import base64
import pickle
from io import BytesIO
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from PIL import Image

# Initialize FastAPI app
app = FastAPI(title="Ficbot Model API", version="1.0")

# Define model paths
MODEL_PATH = "/app/models/img2name.keras"
MAPS_PATH = "/app/models/maps.pkl"

if os.environ.get("TESTING"):
    MODEL_PATH = "src/core/models/img2name/img2name.keras"
    MAPS_PATH = "src/core/models/img2name/maps.pkl"

# Load TensorFlow model on startup
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load character mappings
with open(MAPS_PATH, "rb") as mp:
    char_idx, idx_char = pickle.load(mp)

print("Model and mappings loaded!")

def sample(preds, temperature=1.0):
    """ Helper function to sample an index from a probability array """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def preprocess_image(image_bytes):
    """ Converts base64 image bytes into a processed feature vector. """
    target_size = (224, 224)  # Model expected input size
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def generate_name(image_bytes, min_name_length=2, diversity=1.2,
                  start_token="@", end_token="$", ood_token="?"):
    """ Generates a name using the model inside the container. """
    
    image_features = preprocess_image(image_bytes)
    maxlen = model.get_layer("NAME_INPUT").output.shape[1]

    generated = ""
    name = start_token * maxlen

    while not (generated.endswith(start_token) or generated.endswith(end_token)):
        x_pred_text = np.zeros((1, maxlen, len(idx_char)))
        for t, char in enumerate(name):
            x_pred_text[0, t, char_idx[char]] = 1.0

        preds = model.predict([image_features, x_pred_text], verbose=0)[0]
        next_char = ood_token
        while next_char == ood_token:  # in case next_char is ood token, we sample (and then resample) until it isn't
            next_index = sample(preds, diversity)
            next_char = idx_char[next_index]
        if next_char == end_token and generated.count(' ') < min_name_length - 1:
            next_char = " "

        name = name[1:] + next_char
        generated += next_char

    if generated[-1] in {start_token, end_token}:
        generated = generated[:-1]

    generated = [word.capitalize() for word in generated.split()]
    generated = ' '.join(generated)
    return generated

@app.post("/generate/")
async def generate_character_name(request: Request):
    """ Receives an image, runs inference, and returns a generated name. """
    try:
        body = await request.json()
        image_bytes = base64.b64decode(body["image"])
        diversity = float(body.get("diversity", 1.2))
        min_name_length = int(body.get("min_name_length", 2))

        name = generate_name(image_bytes, min_name_length=min_name_length, diversity=diversity)

        return JSONResponse(content={"success": True, "name": name})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """ Health check endpoint to confirm the API is running. """
    return JSONResponse(content={"status": "OK", "message": "API is running!"})