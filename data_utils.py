import html
import json
import re

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

RUBTSOVA_HEADER = 'id tdate tname ttext ttype trep tfav tstcount tfol tfrien listcount'.split(' ')


urls_regex = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mentions_regex = re.compile(r'@(\w+)')
rt_regex = re.compile(r'\brt\b', re.IGNORECASE)
positive_brace_regex = re.compile(r'\){2,}', re.IGNORECASE)
# negative_brace_regex = re.compile(r'\({2,}', re.IGNORECASE)  # это не столько негатив, сколько неудовлетворение
positive_smiles_regex = re.compile(r':-\)|:\)|;\)|\(:|\(=|:D|:o\)|:]|:3|:c\)|:>|=]|8\)|=\)|:}|:^\)|:-D|8-D|8D|XD|=-D|=D|=-3|=3|\)\)|\^\^|\^_\^|\\o/|\\m/|<3|:\*', re.IGNORECASE)  # nopep8
negative_smiles_regex = re.compile(r';\(|:\(|:\[|:{|\(\(|:\'\(|:\'\[|:c|:с|D:|\):|\)=', re.IGNORECASE)
only_letters_regex = re.compile(r'[^\w ]|\d|_')
one_space_regex = re.compile(r' +')
spaces_regex = re.compile(r'\s+')

# positive_smile_replacement = ' smpositive '
# negative_smile_replacement = ' smnegative '

positive_smile_replacement = ''
negative_smile_replacement = ''


def clean_tweet(tweet):
    text = tweet.lower()

    text = html.unescape(text)

    text = rt_regex.sub('', text)
    text = mentions_regex.sub('', text)
    text = urls_regex.sub('', text)

    text = positive_brace_regex.sub(positive_smile_replacement, text)
    text = positive_smiles_regex.sub(positive_smile_replacement, text)
    text = negative_smiles_regex.sub(negative_smile_replacement, text)

    text = only_letters_regex.sub(' ', text)
    text = one_space_regex.sub(' ', text)

    text = text.strip()
    return text


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

    positive_data = pd.read_csv(positive, header=None, sep=';', index_col=False, names=RUBTSOVA_HEADER)
    negative_data = pd.read_csv(negative, header=None, sep=';', index_col=False, names=RUBTSOVA_HEADER)

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
    dataset = dataset.tolist()
    labels = labels.tolist()

    data = {
        'dataset': dataset,
        'labels': labels,
    }

    with open(to_path, 'w') as f:
        json.dump(data, f, check_circular=False)
