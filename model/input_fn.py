"""data iterator"""

import pickle
import tensorflow as tf
import os


def train_input_fn(data_dir, params):
    """
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    C = pickle.load(open(os.path.join(data_dir, 'train/C.pkl'), 'rb'))  # context
    Q = pickle.load(open(os.path.join(data_dir, 'train/Q.pkl'), 'rb'))  # question
    R = pickle.load(open(os.path.join(data_dir, 'train/R.pkl'), 'rb'))  # response/reply
    I = pickle.load(open(os.path.join(data_dir, 'train/I.pkl'), 'rb'))  # personal Info
    labels = pickle.load(open(os.path.join(data_dir, 'train/Y.pkl'), 'rb'))  # labels

    params.train_size = len(C)

    dataset = tf.data.Dataset.from_tensor_slices((C, Q, R, I, labels))

    dataset = dataset.shuffle(params.train_size)
    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


def eval_input_fn(data_dir, params):
    """
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    C = pickle.load(open(os.path.join(data_dir, 'eval/C.pkl'), 'rb'))  # context
    Q = pickle.load(open(os.path.join(data_dir, 'eval/Q.pkl'), 'rb'))  # question
    R = pickle.load(open(os.path.join(data_dir, 'eval/R.pkl'), 'rb'))  # response/reply
    I = pickle.load(open(os.path.join(data_dir, 'eval/I.pkl'), 'rb'))  # personal Info
    labels = pickle.load(open(os.path.join(data_dir, 'eval/Y.pkl'), 'rb'))  # labels

    dataset = tf.data.Dataset.from_tensor_slices((C, Q, R, I, labels))
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


def final_train_input_fn(data_dir, params):
    """
    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    C = pickle.load(open(os.path.join(data_dir, 'C.pkl'), 'rb'))  # context
    Q = pickle.load(open(os.path.join(data_dir, 'Q.pkl'), 'rb'))  # question
    R = pickle.load(open(os.path.join(data_dir, 'R.pkl'), 'rb'))  # response/reply
    I = pickle.load(open(os.path.join(data_dir, 'I.pkl'), 'rb'))  # personal Info
    labels = pickle.load(open(os.path.join(data_dir, 'Y.pkl'), 'rb'))  # labels

    params.train_size = len(C)

    dataset = tf.data.Dataset.from_tensor_slices((C, Q, R, I, labels))

    dataset = dataset.shuffle(params.train_size)
    dataset = dataset.repeat(params.num_epochs)
    dataset = dataset.batch(params.batch_size)
    dataset = dataset.prefetch(buffer_size=None)
    return dataset
