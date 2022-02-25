import tensorflow as tf
from tensorflow.keras.applications import MobileNet


class SimpleImageNameModel(object):

    def __init__(self, *, maxlen, vocab_size):
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        # Inputs
        self.image_input = tf.keras.Input(shape=(224, 224, 3), name="IMAGE_INPUT")
        self.name_input = tf.keras.Input(shape=(maxlen, vocab_size), name="NAME_INPUT")
        # Transfer
        self.transfer = MobileNet(weights="imagenet", include_top=False, pooling="avg")
        self.transfer.trainable = False
        # LSTM
        self.rnn = tf.keras.layers.LSTM(128)
        # Common
        self.concatenate = tf.keras.layers.Concatenate(axis=1)
        self.predict = tf.keras.layers.Dense(vocab_size, activation="softmax", name="PREDICTION")

    def __new__(cls, *, maxlen, vocab_size):
        model = super(SimpleImageNameModel, cls).__new__(cls)
        model.__init__(maxlen=maxlen, vocab_size=vocab_size)
        return model.__build()

    def __build(self):
        image_features = self.transfer(self.image_input)
        name_features = self.rnn(self.name_input)
        x = self.concatenate([image_features, name_features])
        predictions = self.predict(x)
        model = tf.keras.Model(inputs=[self.image_input, self.name_input], outputs=predictions, name="SimpleImageNameModel")
        return model


if __name__ == "__main__":
    sample_model = SimpleImageNameModel(maxlen=3, vocab_size=420)
    sample_model.compile(loss="categorical_crossentropy", optimizer="adam")
    print(sample_model.summary())
