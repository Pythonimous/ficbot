import argparse
import sys

from src.core import train
from src.core.models.img2name import Img2Name
from src.core import loaders


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


def model_train(model, loader, arguments):
    """ Train a model from scratch """
    create_loader = loaders.create_loader
    train_model = train.train_model

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
    model, loader = train.load_from_checkpoint(checkpoint_path=arguments.checkpoint,
                                               data_path=arguments.data_path,
                                               model_name=arguments.model,
                                               img_dir=arguments.img_dir, batch_size=arguments.batch_size,
                                               maps_path=arguments.maps,
                                               name_col=arguments.name_col, img_col=arguments.img_col)
    train.train_model(model, loader, arguments.checkpoint_dir, epochs=arguments.epochs)


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


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
