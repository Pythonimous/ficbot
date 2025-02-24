import numpy as np
import tensorflow as tf

def sample(preds, temperature=1.0):
    """ Helper function to sample an index from a probability array """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def preprocess_image_array(image_array, target_size=(224, 224), preprocess_for="mobilenet"):
    image_array = tf.image.resize(image_array, (target_size[0], target_size[1])).numpy()
    preprocessing = getattr(tf.keras.applications, preprocess_for, tf.keras.applications.mobilenet)
    return preprocessing.preprocess_input(image_array)
