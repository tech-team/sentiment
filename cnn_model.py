import argparse

import yaml
import tensorflow as tf

from data_utils import load_dataset, text2words, load_rubtsova_datasets
from sentiment.cnn import SentimentCNN


def train(interactive=False, config_file=None):
    return _initialize('train', interactive, config_file)


def load_model(interactive=False, config_file=None):
    return _initialize('load', interactive, config_file)


def _initialize(mode, interactive, config_file):
    if mode not in ['train', 'load']:
        raise Exception('mode should be one of \'train\' or \'load\'')

    config = load_config(config_file)
    cnn_config = config['cnn']
    ds_config = config['datasets']

    train_dataset, train_labels, valid_dataset, valid_labels = None, None, None, None
    if mode == 'train':
        train_dataset, train_labels, valid_dataset, valid_labels = \
            load_rubtsova_datasets(ds_config['positive'],
                                   ds_config['negative'],
                                   ds_config['size'])

        max_len = max(map(len, train_dataset))
        print('Maximum sentence length: {}'.format(max_len))

    with tf.Graph().as_default() as graph:
        session = tf.Session(graph=graph)
        cnn = SentimentCNN(
            session=session,
            **cnn_config
        )

        if mode == 'train':
            cnn.train(train_dataset=train_dataset, train_labels=train_labels,
                      valid_dataset=valid_dataset, valid_labels=valid_labels)
        else:
            cnn.restore()

        if interactive is True:
            run_interactive(cnn)

        return cnn


def load_config(file_name):
    with open(file_name, 'r') as config_file:
        return yaml.load(config_file.read())


def predict(cnn, sentence):
    words = text2words(sentence)
    prediction = cnn.predict(words)

    n = prediction[0]
    p = prediction[1]

    return n, p


def run_interactive(cnn):
    while True:
        try:
            text = input('Text: ')
            n, p = predict(cnn, text)
            print('Negative: {:g}. Positive: {:g}'.format(n, p))
            print()
        except KeyboardInterrupt:
            return
        except:
            pass


def _str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Need bool; got %r' % s)
    return {'true': True, 'false': False}[s.lower()]


def add_boolean_argument(parser, name, default=False):
    """Add a boolean argument to an ArgumentParser instance."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--' + name, nargs='?', default=default, const=True, type=_str_to_bool)
    group.add_argument('--no' + name, dest=name, action='store_false')


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', metavar='M',
                        help='CSV files to process',
                        type=str)

    add_boolean_argument(parser, 'interactive', True)

    parser.add_argument('-c', '--meta_config_file',
                        help='path to meta_config.yml',
                        type=str,
                        required=True)

    args = parser.parse_args()

    _initialize(args.mode, interactive=args.interactive, config_file=args.meta_config_file)


if __name__ == '__main__':
    cli_main()
