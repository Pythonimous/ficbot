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


def get_image(path, target_size, preprocess_for="mobilenet"):

    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)

    image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()
    preprocessing = getattr(tf.keras.applications, preprocess_for, tf.keras.applications.mobilenet)
    image_arr = preprocessing.preprocess_input(image_arr)
    return image_arr