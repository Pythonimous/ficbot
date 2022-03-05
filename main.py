import argparse
import sys

from ficbot.tf_models import train as tf_train
from ficbot.tf_models.name.models import SimpleModel
from ficbot.tf_models.name.generation import generate_name as tf_generate_name
from ficbot.data.loaders import tf_loaders


def get_model_class(framework, model_key):
    models = {
        "tf": {
            "simple_img_name": SimpleModel
        },
        "torch": {
        }
    }
    loaders = {
        "tf": {
            "simple_img_name": "ImgNameLoader"
        },
        "torch": {

            }
    }

    model = models[framework].get(model_key, None)

    if not model:
        sys.exit("Model not yet implemented in this framework! Please choose a different framework.")

    loader = loaders[framework][model_key]
    return model, loader


def get_model(arguments):

    framework = arguments.framework

    if arguments.model:
        model_key = arguments.model
        return get_model_class(framework, model_key)


def choose_model(arguments):

    inputs = list(set(arguments.inputs))
    outputs = list(set(arguments.outputs))

    input_candidates = {
        "name": set(),
        "bio": set(),
        "image": {"simple_img_name"}
    }

    output_candidates = {
        "name": {"simple_img_name"},
        "bio": set(),
        "image": set()
    }

    all_models = ["simple_img_name"]
    candidates = set(all_models)
    candidate_descriptions = {
        "simple_img_name": "Simple img -> name model.\n"
                           "Train: uses char sequenized names + corresponding images as input "
                           "to predict the next character of a name.\n"
                           "Generate: Uses single image as an input."
    }

    for input in inputs:
        candidates.intersection_update(input_candidates[input])
    for output in outputs:
        candidates.intersection_update(output_candidates[output])

    if not candidates:
        print("\nNo model with such capabilities yet :)\nCurrently available models:")
        candidates = all_models
    else:
        print("\nModels for your specifications:")

    for candidate in list(candidates):
        print(f"{candidate}: {candidate_descriptions[candidate]}\n")

    sys.exit()


def model_train_new(model, loader, arguments):

    if arguments.framework == "tf":
        create_loader = tf_loaders.create_loader
        train_model = tf_train.train_model
    else:
        create_loader = tf_loaders.create_loader  # torch_loaders.create_loader
        train_model = tf_train.train_model  # torch_train.train_model

    if arguments.model == "simple_img_name":
        loader = create_loader(arguments.data_path, loader=loader, img_dir=arguments.img_dir,
                               img_col=arguments.img_col, name_col=arguments.name_col,
                               maxlen=arguments.maxlen, batch_size=arguments.batch_size)
        vocab_size = loader.vectorizer.get_vocab_size()
        simple_img_name = model(maxlen=arguments.maxlen, vocab_size=vocab_size)
        simple_img_name.compile(loss="categorical_crossentropy", optimizer=arguments.optimizer)
        train_model(simple_img_name, loader, arguments.checkpoint_dir, epochs=arguments.epochs)


def model_train_checkpoint(arguments):
    if arguments.framework == "tf":
        model, loader = tf_train.load_from_checkpoint(checkpoint_path=arguments.checkpoint,
                                                      data_path=arguments.data_path,
                                                      model_name=arguments.model,
                                                      img_dir=arguments.img_dir, batch_size=arguments.batch_size,
                                                      maps_path=arguments.maps,
                                                      name_col=arguments.name_col, img_col=arguments.img_col)
        tf_train.train_model(model, loader, arguments.checkpoint_dir, epochs=arguments.epochs)


def model_generate(arguments):
    if arguments.framework == "tf":
        # TODO: automatically detect framework. maybe write a common generate_name function for both tf and torch?
        # TODO: there will be more functions and input -> output configurations, so write a more flexible framework here
        name = tf_generate_name(arguments.image_path,
                                arguments.model_path,
                                arguments.maps_path,
                                diversity=arguments.diversity)
        # TODO: add custom start / end / ood token support
    else:
        name = None
    print("Generated name: ", name)


def main(arguments):

    if arguments.inputs:
        choose_model(arguments)

    elif arguments.train:
        if not arguments.checkpoint:
            model, loader = get_model(arguments)
            model_train_new(model, loader, arguments)
        else:
            model_train_checkpoint(arguments)
    else:
        model_generate(arguments)


def parse_arguments():
    parser = argparse.ArgumentParser(prog='Ficbot', description='Your friendly neighborhood fanfic writing assistant! '
                                                                'Boost your imagination with a bit of AI magic.')

    general_group = parser.add_argument_group('General', 'General parameters')
    general_group.add_argument('--train', action='store_true',
                               help='train a new generation model from scratch or checkpoint')
    general_group.add_argument('--framework', choices=['tf', 'torch'],
                               help='model from which framework to use')
    general_group.add_argument('--model', nargs="?",
                               help='model class to use. when in doubt, use --input and --output commands,'
                               'and this script will decide for you.')
    general_group.add_argument('--inputs', choices=['name', 'image', 'bio'],
                               nargs='*', help='inputs the model should accept')
    general_group.add_argument('--outputs', choices=['name', 'image', 'bio'],
                               nargs='*', help='outputs the model should generate')

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

    generate_group = parser.add_argument_group('Generate', 'Generation parameters')
    generate_group.add_argument('--model_path', default='example/name/tf_simple_average.hdf5',
                                help='model path for generation')
    generate_group.add_argument('--maps_path', default='example/name/maps.pkl',
                                help='path to maps for generation (if applicable to a model)')
    generate_group.add_argument('--image_path', default='example/name/example.jpg',
                                help='image path for generation')
    generate_group.add_argument('--diversity', type=float, default=1.0,
                                help='generation sampling diversity, the higher the more diverse')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments)
