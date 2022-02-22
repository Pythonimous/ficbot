import numpy as np
import tensorflow as tf
from ficbot.data.loaders.tf_loaders import ImageLoader

import pickle

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def sample(preds, temperature=1.0):
    """ Helper function to sample an index from a probability array """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_name(image_path, model_path, maps_path, *,
                  diversity: float = 1.2, start_token: str = "@", end_token: str = "$"):
    name_model = tf.keras.models.load_model(model_path)
    maxlen = name_model.get_layer("NAME_INPUT").output_shape[0][1]
    preprocessing_algorithm = ""
    for layer in name_model.layers:
        if layer.name in {"vgg16", "vgg19", "resnet50"}:
            preprocessing_algorithm = layer.name
            break
    image_dim = name_model.get_layer("IMAGE_INPUT").output_shape[0][1:]
    image_features = ImageLoader.get_image(image_path, image_dim, preprocessing_algorithm)
    image_features = np.expand_dims(image_features, axis=0)

    with open(maps_path, 'rb') as mp:
        maps = pickle.load(mp)

    idx_char = maps["idx_char"]
    char_idx = maps["char_idx"]

    generated = ""
    name = start_token * maxlen

    while not generated.endswith(end_token) and len(generated) <= 100:
        x_pred_text = np.zeros((1, maxlen, len(idx_char)))
        for t, char in enumerate(name):
            x_pred_text[0, t, char_idx[char]] = 1.0
        preds = name_model.predict([image_features, x_pred_text], verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = idx_char[next_index]
        name = name[1:] + next_char
        generated += next_char

    return generated


if __name__ == "__main__":
    generated_name = generate_name("../../../tests/test_files/data/tf_loaders/images/example.jpg",
                                   "../../../models/name_generation/tf/checkpoints/simple_untrained.hdf5",
                                   "../../../models/name_generation/tf/checkpoints/name_maps.pkl")
    print(generated_name)
