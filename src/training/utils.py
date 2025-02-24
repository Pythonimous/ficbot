import tensorflow as tf


def get_image(path, target_size, preprocess_for="mobilenet"):

    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)

    image_arr = tf.image.resize(image_arr, (target_size[0], target_size[1])).numpy()
    preprocessing = getattr(tf.keras.applications, preprocess_for, tf.keras.applications.mobilenet)
    image_arr = preprocessing.preprocess_input(image_arr)
    return image_arr