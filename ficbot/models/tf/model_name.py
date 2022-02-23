import tensorflow as tf
from tensorflow.keras.applications import MobileNet


def simple_image_name_model(maxlen, vocab_size, *, loss, optimizer):
    image_input = tf.keras.Input(shape=(224, 224, 3), name="IMAGE_INPUT")

    transfer_layer = MobileNet(weights="imagenet", include_top=False, pooling='avg')
    transfer_layer.trainable = False  # freeze weights for feature extraction

    image_features = transfer_layer(image_input)
    name_input = tf.keras.Input(shape=(maxlen, vocab_size), name="NAME_INPUT")
    rnn = tf.keras.layers.LSTM(128)(name_input)
    concatenated_features = tf.keras.layers.Concatenate(axis=1)([image_features, rnn])
    output = tf.keras.layers.Dense(vocab_size, activation="softmax")(concatenated_features)
    model = tf.keras.Model(inputs=[image_input, name_input], outputs=output, name="SimpleImageNameModel")
    model.compile(loss=loss, optimizer=optimizer)
    return model


if __name__ == "__main__":
    sample_model = simple_image_name_model(3, 420, loss='categorical_crossentropy', optimizer='adam')
    print(sample_model.summary())
