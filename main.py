import re

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

from sentiment.cnn import SentimentCNN


def make_one_hot(arr, size=None):
    if size is None:
        size = np.max(arr) + 1
    return np.eye(size)[arr]


def load_dataset(path):
    dataset = []
    labels = []
    data = pd.read_csv(path, header=None, encoding='latin1')
    data = data.reindex(np.random.permutation(data.index))
    data_len = len(data)
    print('read_csv finished. data_len = {}'.format(data_len))

    letter_emoticons_regex = re.compile(r'[:;][dpoзv]', re.IGNORECASE)
    only_letters_regex = re.compile(r'[^\w ]|\d', re.IGNORECASE)
    urls_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                            re.IGNORECASE)
    mentions_regex = re.compile(r'@(\w+)', re.IGNORECASE)
    spaces_regex = re.compile(r'\s+')

    i = 0
    for row_id, row in data.iterrows():
        sentiment = row[0]
        text = row[5]
        if sentiment != 2:
            text = text.lower()
            text = letter_emoticons_regex.sub('', text)
            text = mentions_regex.sub('', text)
            text = urls_regex.sub('', text)
            text = only_letters_regex.sub('', text)
            text = text.strip()
            words = spaces_regex.split(text)
            dataset.append(words)

            labels.append(0 if sentiment == 0 else 1)
            i += 1
        if i >= 100000:
            break
    dataset = np.asarray(dataset)
    labels = make_one_hot(labels, size=2)
    print('Finished loading dataset')
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=42)
    return X_train, y_train, X_test, y_test


def main():
    train_dataset, train_labels, valid_dataset, valid_labels = \
        load_dataset('data/training.1600000.processed.noemoticon.csv')
    with tf.Session() as session:
        cnn = SentimentCNN(
            session=session,
            embeddings_model_path='./sentiment/saved/model.ckpt-2264733',
            embeddings_vocab_path='./sentiment/saved/vocab.txt',
            embeddings_size=200,
            sentence_length=60,
            n_labels=2,
            filter_sizes=(2, 3, 4),
            n_filters=2,
            n_steps=100000
        )

        cnn.train(train_dataset=train_dataset, train_labels=train_labels,
                  valid_dataset=valid_dataset, valid_labels=valid_labels)

if __name__ == '__main__':
    main()
