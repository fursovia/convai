"""tokenize and vectorize"""

import os
import argparse
import pandas as pd
import tensorflow as tf
import pickle
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/data/i.anokhin/convai/data_convai_string', help="Directory containing the raw dataset")
parser.add_argument('--nrows', type=int, default=-1)


if __name__ == '__main__':

    args = parser.parse_args()
    table_path = os.path.join(args.data_dir, 'labeled_char_df.csv')
    print()

    assert os.path.isfile(table_path), 'No files found at {}'.format(table_path)

    if args.nrows != -1:
        df_cleaned = pd.read_csv(table_path, nrows=args.nrows)
    else:
        df_cleaned = pd.read_csv(table_path, nrows=None)

    df_cleaned = df_cleaned.fillna('')
    # df_cleaned = df_cleaned[:2000]

    print(df_cleaned.columns)
    print(df_cleaned.shape)

    Y = df_cleaned['labels'].values.ravel()
    responses = df_cleaned['reply'].values.ravel()

    sample_path = os.path.join(args.data_dir, 'sample')

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    Ytr, Yev, Xtr, Xev, Rtr, Rte = train_test_split(Y,
                                                    df_cleaned,
                                                    responses,
                                                    test_size=0.1,
                                                    random_state=24)

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def convert_to_records(data, name, save_to):
        """Converts a dataset to tfrecords."""
        print(data.shape)
        num_examples = data.shape[0]

        filename = os.path.join(save_to, name + '.tfrecords')
        print('Writing...', filename)

        # print('data', data)

        with tf.python_io.TFRecordWriter(filename) as writer:
            for index in range(num_examples):
                Y_ = df_cleaned['labels'][index]
                cont = df_cleaned['context'][index].encode()
                quest = df_cleaned['question'][index].encode()
                resp = df_cleaned['reply'][index].encode()
                fact1 = df_cleaned['fact1'][index].encode()
                fact2 = df_cleaned['fact2'][index].encode()
                fact3 = df_cleaned['fact3'][index].encode()
                fact4 = df_cleaned['fact4'][index].encode()
                fact5 = df_cleaned['fact5'][index].encode()

                # print('question', quest)

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'label': _int64_feature([Y_]),
                            'cont': _bytes_feature(cont),
                            'quest': _bytes_feature(quest),
                            'resp': _bytes_feature(resp),
                            'fact1': _bytes_feature(fact1),
                            'fact2': _bytes_feature(fact2),
                            'fact3': _bytes_feature(fact3),
                            'fact4': _bytes_feature(fact4),
                            'fact5': _bytes_feature(fact5),
                        }
                    )
                )
                writer.write(example.SerializeToString())

    print('Converting to TFRecords...')
    convert_to_records(df_cleaned, 'full', args.data_dir)
    convert_to_records(Xtr, 'train', args.data_dir)
    convert_to_records(Xev, 'eval', args.data_dir)

