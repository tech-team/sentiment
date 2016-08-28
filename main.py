import argparse

import tensorflow as tf

from data_utils import load_dataset, text2words, load_rubtsova_datasets
from sentiment.cnn import SentimentCNN


def main(mode, max_len=None):
    if mode not in ['train', 'load']:
        raise Exception('mode should be one of \'train\' or \'load\'')

    train_dataset, train_labels, valid_dataset, valid_labels = None, None, None, None
    if mode == 'train':
        train_dataset, train_labels, valid_dataset, valid_labels = \
            load_rubtsova_datasets('data/rubtsova/positive.csv',
                                   'data/rubtsova/negative.csv',
                                   size=10000)

        max_len = max(map(len, train_dataset))
        print('Maximum sentence length: {}'.format(max_len))

    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as session:
            cnn = SentimentCNN(
                session=session,
                embeddings_model_path='./preprocess/sql_utils/trained/model.ckpt-27138193',
                embeddings_vocab_path='./preprocess/sql_utils/trained/vocab.txt',
                model_save_path='./cnn_saved/',
                embeddings_size=200,
                sentence_length=max_len,
                n_labels=2,
                filter_sizes=(3, 4, 5),
                dropout_keep_prob=0.4,
                l2_lambda=0.1,
                n_filters=128,
                n_steps=20,
                batch_size=256,
                learning_rate=0.002,
                validation_check_steps=100,
                summary_path='./summary'
            )

            if mode == 'train':
                cnn.train(train_dataset=train_dataset, train_labels=train_labels,
                          valid_dataset=valid_dataset, valid_labels=valid_labels)
            else:
                cnn.restore()

            run_interactive(cnn)


def run_interactive(cnn):
    while True:
        try:
            text = input('Text: ')
            words = text2words(text)
            prediction = cnn.predict(words)

            n = prediction[0]
            p = prediction[1]

            print('Negative: {:g}. Positive: {:g}'.format(n, p))
            print()
        except KeyboardInterrupt:
            return
        except:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', metavar='M',
                        help='CSV files to process',
                        type=str)

    # TODO: we really don't want to load train sets for eagle, but we need to somehow get max_len
    parser.add_argument('-l', '--max_len',
                        help='max len of sentence in train set (only for load mode)',
                        type=int,
                        required=False)
    args = parser.parse_args()

    main(args.mode, args.max_len)
