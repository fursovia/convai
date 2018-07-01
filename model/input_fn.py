import tensorflow as tf
import os
from model.utils import decode


def input_fn(data_dir, params, file_name, train_time=True, evaluate_epochs=None):
    dataset = tf.data.TFRecordDataset(os.path.join(data_dir, '{}.tfrecords'.format(file_name)))

    if train_time:
        if evaluate_epochs is not None:
            num_epochs = evaluate_epochs
        else:
            num_epochs = params.num_epochs

        dataset = dataset.shuffle(params.train_size)
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.apply(tf.contrib.data.map_and_batch(decode, params.batch_size, num_parallel_batches=4))
    dataset = dataset.prefetch(buffer_size=None)
    return dataset
