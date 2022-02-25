from ficbot.models.name_gen.tf.build_model import SimpleImageNameModel
from ficbot.data.loaders.tf_loaders import create_loader
from ficbot.features.vectorizer import SequenceVectorizer
import tensorflow as tf
import os
import time

import pickle


def create_simple_name_model(*, maxlen, vocab_size, loss, optimizer):

    model = SimpleImageNameModel(maxlen=maxlen, vocab_size=vocab_size)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def load_from_checkpoint(*, checkpoint_path, data_path, model_type, loader_type, **kwargs):

    model_types = {"img-name"}
    assert model_type in model_types, f"No such model yet.\nAvailable models: {', '.join(list(model_types))}."

    model = tf.keras.models.load_model(checkpoint_path)

    if model_type == "img-name":
        with open(kwargs["maps_path"], 'rb') as mp:
            maps = pickle.load(mp)
        vectorizer = SequenceVectorizer(maps=maps)

        preprocessing_algorithm = "mobilenet"
        for layer in model.layers:
            if layer.name in {"vgg16", "vgg19", "resnet50", "mobilenet"}:
                preprocessing_algorithm = layer.name
                break

        maxlen = model.get_layer("NAME_INPUT").output_shape[0][1]

        loader = create_loader(data_path, loader=loader_type,
                               vectorizer=vectorizer,
                               transfer_net=preprocessing_algorithm,
                               img_folder=kwargs["img_folder"],
                               img_col=kwargs["img_col"],
                               name_col=kwargs["name_col"],
                               maxlen=maxlen, batch_size=kwargs["batch_size"])
    else:
        loader = None

    return model, loader


def train_model(model, loader, checkpoint_folder, *, epochs: int = 1):

    loader.vectorizer.save_maps(checkpoint_folder)

    checkpoint_path = os.path.join(checkpoint_folder, "simple.{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='loss',
                                                    verbose=1, save_best_only=False,
                                                    save_weights_only=False, mode='min')
    callbacks = [checkpoint]
    model.fit(loader, epochs=epochs, callbacks=callbacks, verbose=1)


if __name__ == "__main__":

    checkpoint_folder = os.path.join("../../../../models/name_generation/tf/checkpoints", str(int(time.time())))

    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    loader = create_loader("../../../../data/interim/img_name.csv",
                           loader="ImgNameLoader",
                           img_folder="../../../../data/raw/images",
                           img_col="image", name_col="eng_name",
                           maxlen=3, batch_size=1)

    loader.vectorizer.save_maps(checkpoint_folder)

    maxlen = loader.get_maxlen()
    vocab_size = loader.vectorizer.get_vocab_size()

    model = create_simple_name_model(maxlen=maxlen, vocab_size=vocab_size,
                                     loss="categorical_crossentropy", optimizer="adam")
    train_model(model, loader, checkpoint_folder, epochs=1)
