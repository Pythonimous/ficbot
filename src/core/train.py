from src.core.loaders import create_loader
from src.features.vectorizer import SequenceVectorizer
import tensorflow as tf
import os
import time

import pickle


def load_from_checkpoint(*, checkpoint_path, data_path, model_name, **kwargs):

    model = tf.keras.models.load_model(checkpoint_path)

    if "name" in model_name:
        with open(kwargs["maps_path"], 'rb') as mp:
            maps = pickle.load(mp)
        kwargs["vectorizer"] = SequenceVectorizer(maps=maps)
        kwargs["maxlen"] = model.get_layer("NAME_INPUT").output_shape[0][1]

    if "image" in model_name:
        for layer in model.layers:
            if layer.name in {"vgg16", "vgg19", "resnet50", "mobilenet"}:
                kwargs["transfer_net"] = layer.name
                break

    loader = create_loader(data_path, load_for=model_name, **kwargs)

    return model, loader


def train_model(model, loader, checkpoint_folder, *, epochs: int = 1):

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')  # Use the first GPU
            print("Using GPU:", gpus[0])
        except RuntimeError as e:
            print(e)

    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    checkpoint_folder = os.path.join(checkpoint_folder, str(int(time.time())))

    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    loader.vectorizer.save_maps(checkpoint_folder)

    checkpoint_path = os.path.join(checkpoint_folder, "simple.{epoch:02d}-{loss:.2f}.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss',
                                                    verbose=1, save_best_only=False,
                                                    save_weights_only=False, mode='min')
    callbacks = [checkpoint]
    model.fit(loader, epochs=epochs, callbacks=callbacks, verbose=1)


if __name__ == "__main__":

    checkpoint_folder = os.path.join("../../models/name_generation/tf/checkpoints", str(int(time.time())))

    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    model, loader = load_from_checkpoint(
        maps_path="../../example/name/maps.pkl",
        checkpoint_path="../../example/name/tf_simple_average.hdf5",
        data_path="../../data/interim/img_name.csv",
        model_name="simple_img_name",
        img_dir="../../data/raw/images",
        img_col="image", name_col="eng_name",
        batch_size=1)

    train_model(model, loader, checkpoint_folder, epochs=6)
