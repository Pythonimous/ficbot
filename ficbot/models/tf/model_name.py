import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


def simple_image_name_model(maxlen, vocab_size, *, loss, optimizer, colab_tpu=False):
    image_input = tf.keras.Input(shape=(224, 224, 3), name="IMAGE_INPUT")
    image_features = VGG16(weights="imagenet", include_top=False, pooling='avg')(image_input)
    name_input = tf.keras.Input(shape=(maxlen, vocab_size), name="NAME_INPUT")
    lstm = tf.keras.layers.LSTM(128)(name_input)
    concatenated_features = tf.keras.layers.Concatenate(axis=1)([image_features, lstm])
    output = tf.keras.layers.Dense(vocab_size, activation="softmax")(concatenated_features)
    model = tf.keras.Model(inputs=[image_input, name_input], outputs=output, name="SimpleImageNameModel")
    if colab_tpu:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])

        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)

        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("REPLICAS: ", strategy.num_replicas_in_sync)
        with strategy.scope():
            model.compile(loss=loss, optimizer=optimizer)
    return model


if __name__ == "__main__":
    sample_model = simple_image_name_model(3, 420, loss='categorical_crossentropy', optimizer='adam')
    print(sample_model.summary())
