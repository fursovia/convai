import tensorflow as tf
import os
from model.utils import decode
import pandas as pd


def input_fn(data_dir, params, file_name, train_time=True, evaluate_epochs=None):
    dataset = tf.data.TFRecordDataset(os.path.join(data_dir, '{}.tfrecords'.format(file_name)))

    if train_time:
        if evaluate_epochs is not None:
            num_epochs = evaluate_epochs
        else:
            num_epochs = params.num_epochs

        dataset = dataset.shuffle(params.train_size)
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.apply(tf.contrib.data.map_and_batch(decode, params.batch_size, num_parallel_batches=40))
    dataset = dataset.prefetch(buffer_size=None)
    return dataset


def input_fn2(data_dir, params, file_name, train_time=True, evaluate_epochs=None):

    data = pd.read_csv(os.path.join(data_dir, '{}.csv'.format(file_name)))
    data = data.fillna('')

    if train_time:
        if evaluate_epochs is not None:
            num_epochs = evaluate_epochs
        else:
            num_epochs = params.num_epochs

        train_input = tf.estimator.inputs.pandas_input_fn(
            data,
            data["labels"],
            num_epochs=num_epochs,
            shuffle=True)
    else:
        train_input = tf.estimator.inputs.pandas_input_fn(
            data,
            data["labels"],
            num_epochs=1,
            shuffle=False)

    return train_input


def pred_input_fn(df):

    data = df
    train_input = tf.estimator.inputs.pandas_input_fn(
            data,
            num_epochs=1,
            shuffle=False)

    return train_input