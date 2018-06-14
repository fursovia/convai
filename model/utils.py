"""Вспомогательные функции"""

import json
import os
import numpy as np
import pickle


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


def get_coefs(word, *arr):
    return word, np.array(arr, dtype="float32")


def get_embeddings(params):
    word2idx_file = os.path.join(params.vocab_path, 'word2idx.pkl')
    embedding_file = os.path.join(params.vocab_path, 'fasttext.vec')

    word2idx = pickle.load(open(word2idx_file, 'rb'))
    embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(embedding_file, encoding='utf-8'))

    embedding_matrix = np.zeros(((params.vocab_size + 1), params.embedding_size))
    for word, i in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(-0.10, 0.10, params.embedding_size)

    return embedding_matrix
