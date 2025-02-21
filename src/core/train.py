import argparse
import sys
import os
import time
import pickle

import tensorflow as tf

from src.core.loaders import create_loader
from src.core.models.img2name import Img2Name

from src.features.vectorizer import SequenceVectorizer


def get_model_class(model_key):
    models = {
        "simple_img_name": Img2Name
    }
    loaders = {
        "simple_img_name": "ImgNameLoader"
    }

    model = models.get(model_key, None)

    if not model:
        sys.exit("Such model doesn't exist!")

    loader = loaders[model_key]
    return model, loader


def load_from_checkpoint(*, checkpoint_path, data_path, model_name, **kwargs):

    model = tf.keras.models.load_model(checkpoint_path)

    if "name" in model_name:
        with open(kwargs["maps_path"], 'rb') as mp:
            maps = pickle.load(mp)
        kwargs["vectorizer"] = SequenceVectorizer(maps=maps)
        kwargs["maxlen"] = model.get_layer("NAME_INPUT").output.shape[1]

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


def model_train(model, loader, arguments):
    """ Train a model from scratch """

    if arguments.model == "simple_img_name":
        loader = create_loader(arguments.data_path, loader=loader, load_for=arguments.model,
                               img_dir=arguments.img_dir,
                               img_col=arguments.img_col, name_col=arguments.name_col,
                               maxlen=arguments.maxlen, batch_size=arguments.batch_size)
        vocab_size = loader.vectorizer.get_vocab_size()
        simple_img_name = model(maxlen=arguments.maxlen, vocab_size=vocab_size)
        simple_img_name.compile(loss="categorical_crossentropy", optimizer=arguments.optimizer)
        train_model(simple_img_name, loader, arguments.checkpoint_dir, epochs=arguments.epochs)


def model_train_checkpoint(arguments):
    """ Train a model from checkpoint """
    model, loader = load_from_checkpoint(checkpoint_path=arguments.checkpoint,
                                               data_path=arguments.data_path,
                                               model_name=arguments.model,
                                               img_dir=arguments.img_dir, batch_size=arguments.batch_size,
                                               maps_path=arguments.maps,
                                               name_col=arguments.name_col, img_col=arguments.img_col)
    train_model(model, loader, arguments.checkpoint_dir, epochs=arguments.epochs)


def parse_arguments():
    parser = argparse.ArgumentParser(prog='Ficbot', description='Your friendly neighborhood fanfic writing assistant! '
                                                                'Boost your imagination with a bit of AI magic.')
    
    parser.add_argument('--info', action='store_true', help='show models available for training')

    train_group = parser.add_argument_group('Train', 'Training parameters')

    train_group.add_argument('--data_path', default='data/interim/img_name.csv', metavar='DATA_PATH',
                             help='the location of your tabular data')
    train_group.add_argument('--name_col', nargs='?', help='name column in your dataframe')
    train_group.add_argument('--bio_col', nargs='?', help='biography column in your dataframe')
    train_group.add_argument('--img_col', nargs='?', help='image file name column in your dataframe')
    train_group.add_argument('--img_dir', default='data/raw/images', nargs='?', metavar='IMG_PATH',
                             help='the location of your images')
    train_group.add_argument('--checkpoint', nargs='?', metavar='CHKP', help='path to checkpoint to train from')
    train_group.add_argument('--maps', nargs='?', help='path to maps for vectorizer (if applicable to model)')
    train_group.add_argument('--checkpoint_dir', default='checkpoints/',
                             help='directory where the checkpoints will be saved')
    train_group.add_argument('--batch_size', type=int, default=16,
                             help='batch size for training the model')
    train_group.add_argument('--epochs', type=int, default=1,
                             help='how long to train the model for')
    train_group.add_argument('--maxlen', type=int, default=3,
                             nargs='?', help='max sequence length for sequence-based generation')
    train_group.add_argument('--optimizer', default='adam', choices=['adam', 'rmsprop'],
                             help='optimizer to use during training')
    
    args = parser.parse_args()
    return args


def main(arguments):

    if arguments.info:
        print("Models available for training:")
        print("simple_img_name: Image to name model")
        print("Good luck!")
        sys.exit()

    if not arguments.checkpoint:
        model, loader = get_model_class(arguments.model)
        model_train(model, loader, arguments)
    else:
        model_train_checkpoint(arguments)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
