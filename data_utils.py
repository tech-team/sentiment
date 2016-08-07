import json

import re

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

from preprocess.rubtsova_csv_to_corpus import clean_tweet

# deprecated: use regex'es from rubtsova_csv_to_corpus.py instead
letter_emoticons_regex = re.compile(r'[:;][dpoзv]', re.IGNORECASE)
only_letters_regex = re.compile(r'[^\w ]|\d', re.IGNORECASE)
urls_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                        re.IGNORECASE)
mentions_regex = re.compile(r'@(\w+)', re.IGNORECASE)
spaces_regex = re.compile(r'\s+')


def make_one_hot(arr, size=None):
    if size is None:
        size = np.max(arr) + 1
    return np.eye(size)[arr]


def text2words(text):
    text = clean_tweet(text)
    words = spaces_regex.split(text)
    return words


def preprocess_dataset(path, size=None):
    dataset = []
    labels = []
    data = pd.read_csv(path, header=None, encoding='latin1')
    data = data.reindex(np.random.permutation(data.index))
    data_len = len(data)
    print('read_csv finished. data_len = {}'.format(data_len))

    i = 0
    for row_id, row in data.iterrows():
        sentiment = row[0]
        text = row[5]
        if sentiment != 2:
            words = text2words(text)
            dataset.append(words)
            labels.append(0 if sentiment == 0 else 1)
            i += 1

        if size and i >= size:
            break

    dataset = np.asarray(dataset)
    labels = make_one_hot(labels, size=2)

    return dataset, labels


def load_preprocessed_dataset(path, size=None):
    with open(path, 'r') as f:
        json_str = f.read()
        data = json.loads(json_str)

    if size is None:
        size = len(data['dataset'])

    return np.asarray(data['dataset'][:size]), np.asarray(data['labels'][:size])


def load_dataset(path, preprocess=False, size=None, shuffle=False):
    if preprocess:
        dataset, labels = preprocess_dataset(path=path, size=size)
    else:
        dataset, labels = load_preprocessed_dataset(path=path, size=size)

    print('Finished loading dataset')
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=42)

    return X_train, y_train, X_test, y_test


def load_rubtsova_datasets(positive, negative, size=None):
    dataset = []
    labels = []

    positive_data = pd.read_csv(positive, header=None, sep=';', index_col=False, names=[
        'id', 'tdate', 'tname', 'ttext', 'ttype',
        'trep', 'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount'])

    negative_data = pd.read_csv(negative, header=None, sep=';', index_col=False, names=[
        'id', 'tdate', 'tname', 'ttext', 'ttype',
        'trep', 'tfav', 'tstcount', 'tfol', 'tfrien', 'listcount'])

    i = 0
    for tweet in positive_data['ttext']:
        words = text2words(tweet)
        dataset.append(words)
        labels.append(1)
        i += 1

        if size and i >= size:
            break

    i = 0
    for tweet in negative_data['ttext']:
        words = text2words(tweet)
        dataset.append(words)
        labels.append(0)
        i += 1

        if size and i >= size:
            break

    dataset = np.asarray(dataset)
    labels = make_one_hot(labels, size=2)

    dataset, labels = shuffle_in_unison(dataset, labels)

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.33, random_state=42)

    return X_train, y_train, X_test, y_test


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

    return a, b


def save_dataset(from_path, to_path, size=None):
    dataset, labels = preprocess_dataset(path=from_path, size=size)

    data = {
        'dataset': dataset.tolist(),
        'labels': labels.tolist(),
    }

    data_str = json.dumps(data, check_circular=False)
    with open(to_path, 'w') as f:
        f.write(data_str)
