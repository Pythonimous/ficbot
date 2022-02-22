import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


def simple_image_name_model(maxlen, vocab_size):
    image_input = tf.keras.Input(shape=(224, 224, 3))
    image_features = VGG16(weights="imagenet", include_top=False, pooling='avg')(image_input)
    name_input = tf.keras.Input(shape=(maxlen, vocab_size))
    lstm = tf.keras.layers.LSTM(128)(name_input)
    concatenated_features = tf.keras.layers.Concatenate(axis=1)([image_features, lstm])
    output = tf.keras.layers.Dense(vocab_size, activation="softmax")(concatenated_features)
    model = tf.keras.Model(inputs=[image_input, name_input], outputs=output, name="SimpleImageNameModel")
    return model


if __name__ == "__main__":
    model = simple_image_name_model(3, 420)
    print(model.summary())
