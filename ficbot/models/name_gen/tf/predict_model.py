import numpy as np
import tensorflow as tf
from ficbot.data.loaders.tf_loaders import ImageLoader

import pickle


def sample(preds, temperature=1.0):
    """ Helper function to sample an index from a probability array """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def load_image_for_model(image_path, model):
    preprocessing_algorithm = ""
    for layer in model.layers:
        if layer.name in {"vgg16", "vgg19", "resnet50", "mobilenet"}:
            preprocessing_algorithm = layer.name
            break
    image_dim = model.get_layer("IMAGE_INPUT").output_shape[0][1:]
    image_features = ImageLoader.get_image(image_path, image_dim, preprocessing_algorithm)
    image_features = np.expand_dims(image_features, axis=0)
    return image_features


def generate_name(image_path, model_path, maps_path, *,
                  diversity: float = 1.2, start_token: str = "@", end_token: str = "$", ood_token: str = "?"):
    name_model = tf.keras.models.load_model(model_path)

    image_features = load_image_for_model(image_path, name_model)

    maxlen = name_model.get_layer("NAME_INPUT").output_shape[0][1]

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

        name = name[1:] + next_char
        generated += next_char

    if generated[-1] in {start_token, end_token}:
        generated = generated[:-1]

    generated = [word.capitalize() for word in generated.split()]
    generated = ' '.join(generated)
    return generated


if __name__ == "__main__":
    generated_name = generate_name("../../../../example/name/example.jpg",
                                   "../../../../example/name/tf_simple_average.hdf5",
                                   "../../../../example/name/maps.pkl",
                                   diversity=1)
    print(generated_name)
