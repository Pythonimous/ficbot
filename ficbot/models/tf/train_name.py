from ficbot.models.tf.model_name import simple_image_name_model
from ficbot.data.loaders.tf_loaders import ImgNameLoader
import tensorflow as tf
import os

import pandas as pd
import pickle


def train_simple_model(data_path, img_folder, checkpoint_folder, img_col, name_col,
                       *, maxlen, loss, optimizer, epochs, batch_size: int = 1, colab_tpu=False):
    mal_data = pd.read_csv(data_path)
    loader = ImgNameLoader(mal_data, img_col, name_col,
                           img_folder=img_folder,
                           maxlen=maxlen,
                           batch_size=batch_size)
    vocab_size = loader.vectorizer.get_vocab_size()

    char_idx, idx_char = loader.vectorizer.get_maps()
    maps = {"char_idx": char_idx, "idx_char": idx_char}
    map_path = os.path.join(checkpoint_folder, "name_maps.pkl")
    with open(map_path, "wb") as mp:
        pickle.dump(maps, mp)

    generator_model = simple_image_name_model(maxlen, vocab_size,
                                              loss=loss, optimizer=optimizer)

    checkpoint_path = os.path.join(checkpoint_folder, "simple_best_weights.hdf5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss',
                                                    verbose=1, save_best_only=True,
                                                    save_weights_only=False, mode='min')
    callbacks = [checkpoint]

    untrained_model_path = os.path.join(checkpoint_folder, f"simple_untrained.hdf5")
    generator_model.save(untrained_model_path)
    if colab_tpu:
        TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        tf.logging.set_verbosity(tf.logging.INFO)

        generator_model = tf.contrib.tpu.keras_to_tpu_model(
            generator_model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
    generator_model.fit(loader, epochs=epochs, callbacks=callbacks, verbose=1)

    return generator_model


if __name__ == "__main__":

    model = train_simple_model("../../../data/interim/img_name.csv",
                               "../../../data/raw/images",
                               "../../../models/name_generation/tf/checkpoints",
                               "image", "eng_name",
                               maxlen=3, epochs=1,
                               loss="categorical_crossentropy",
                               optimizer="adam",
                               batch_size=1)
