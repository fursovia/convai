"""Вспомогательные функции"""

import json
import os
import numpy as np
import pickle
import fasttext
import tensorflow as tf
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import ngrams
from keras.preprocessing.sequence import pad_sequences


snowball_stemmer = SnowballStemmer("english")
stop_words = stopwords.words("english")


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.num_epochs = None
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def get_embeddings(params):
    word2idx_file = os.path.join(params.data_path, 'word2idx.pkl')
    fasttext_file = os.path.join(params.data_path, 'fasttext.bin')

    word2idx = pickle.load(open(word2idx_file, 'rb'))
    model = fasttext.load_model(fasttext_file)

    embedding_matrix = np.zeros(((params.vocab_size + 1), params.embedding_size))

    for word, i in word2idx.items():
        if type(word) == tuple:
            word = ' '.join(word)
        embedding_matrix[i] = model[word]

    return embedding_matrix


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_records(data, name, save_to):
    """Converts a dataset to tfrecords."""
    X, Y = data
    num_examples = Y.shape[0]

    filename = os.path.join(save_to, name + '.tfrecords')
    print('Writing...', filename)

    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
            Y_ = Y[index]
            X_ = X[index]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': _int64_feature([Y_]),
                        'sent': _int64_feature(X_)
                    }
                )
            )
            writer.write(example.SerializeToString())


def decode(serialized_example):
    """Parses an image and label from the given serialized_example."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'sent': tf.FixedLenFeature([200], tf.int64)
        })

    label = tf.cast(features['label'], tf.int64)
    sent = tf.cast(features['sent'], tf.int64)

    return sent, label


def clean(text, with_stopwords=True):
    text = text.strip().lower()
    text = re.sub('[^\w\s]', ' ', text)
    text = re.sub('^\d+\s|\s\d+\s|\s\d+$', " <num> ", text)
    if with_stopwords:
        return ' '.join(snowball_stemmer.stem(word) for word in text.split())
    else:
        return ' '.join(snowball_stemmer.stem(word) for word in text.split() if word not in stop_words)


def vectorize_text(text, word2idx, maxlen=20, truncating_type='post'):
    vec_seen = []

    words_ = text.split()
    bi_ = ngrams(words_, 2)

    for word in words_:
        try:
            vec_seen.append(word2idx[word])
        except:
            continue
    for bi in bi_:
        try:
            vec_seen.append(word2idx[bi])
        except:
            continue
    return pad_sequences([vec_seen], maxlen=maxlen, truncating=truncating_type)[0]


def text2vec(text, word2idx, dict_from_parlai):


    text = clean(text)
    return vectorize_text(text, word2idx)
