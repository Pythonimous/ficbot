import pickle

import numpy as np
import tensorflow as tf

from src.core.loaders import ImageLoader
from src.core.utils import sample, get_image


def load_image_for_model(image_path, model):
    """ Load an image for a particular model and convert it into feature vector """
    preprocessing_algorithm = ""
    for layer in model.layers:
        if layer.name in {"vgg16", "vgg19", "resnet50", "mobilenet"}:
            preprocessing_algorithm = layer.name
            break
    image_dim = model.get_layer("IMAGE_INPUT").output.shape[1:]
    image_features = get_image(image_path, image_dim, preprocessing_algorithm)
    image_features = np.expand_dims(image_features, axis=0)
    return image_features


def generate_name(image_path, model_path, maps_path, *,
                  min_name_length: int = 2, diversity: float = 1.2,
                  start_token: str = "@", end_token: str = "$", ood_token: str = "?"):
    """ Generate a name from an image using a pretrained model """
    name_model = tf.keras.models.load_model(model_path)

    image_features = load_image_for_model(image_path, name_model)

    maxlen = name_model.get_layer("NAME_INPUT").output.shape[1]

    with open(maps_path, 'rb') as mp:
        maps = pickle.load(mp)

    char_idx, idx_char = maps

    generated = ""
    name = start_token * maxlen

    while not (generated.endswith(start_token) or generated.endswith(end_token)):
        x_pred_text = np.zeros((1, maxlen, len(idx_char)))
        for t, char in enumerate(name):
            x_pred_text[0, t, char_idx[char]] = 1.0

        preds = name_model.predict([image_features, x_pred_text], verbose=0)[0]
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
