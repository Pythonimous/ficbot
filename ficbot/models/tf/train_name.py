from ficbot.models.tf.model_name import simple_image_name_model
from ficbot.data.loaders.tf_loaders import ImgNameLoader
import tensorflow as tf

import pandas as pd


def train_model(data_path, img_folder, img_col, name_col, *,
                maxlen, epochs, batch_size: int = 1):
    mal_data = pd.read_csv(data_path)
    loader = ImgNameLoader(mal_data, img_col, name_col,
                           img_folder=img_folder,
                           maxlen=maxlen,
                           batch_size=batch_size)
    vocab_size = loader.vectorizer.get_vocab_size()

    generator_model = simple_image_name_model(maxlen, vocab_size)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    generator_model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    for epoch in range(epochs):
        generator_model.fit(loader, epochs=1)
        return generator_model


if __name__ == "__main__":

    model = train_model("../../../data/interim/img_name.csv",
                        "../../../data/raw/images",
                        "image", "eng_name",
                        maxlen=3, epochs=10,
                        batch_size=2)
