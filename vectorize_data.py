"""tokenize and vectorize"""

import pickle
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from model.utils import convert_to_records
from model.utils import vectorize_chars, vectorize_uni_bi
import pandas as pd
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--nrows', type=int, default=-1)


if __name__ == '__main__':

    args = parser.parse_args()

    raw_path = os.path.join(args.data_dir, 'raw_df.csv')
    table_path = os.path.join(args.data_dir, 'cleaned_df.csv')
    table_path2 = os.path.join(args.data_dir, 'cleaned_char_df.csv')

    assert os.path.isfile(table_path) and os.path.isfile(table_path2), 'No files found at {}'.format(table_path)

    if args.nrows != -1:
        df_cleaned_char = pd.read_csv(table_path2, nrows=args.nrows)
        df_cleaned = pd.read_csv(table_path, nrows=args.nrows)
        df_raw = pd.read_csv(raw_path, nrows=args.nrows)
    else:
        df_cleaned_char = pd.read_csv(table_path2, nrows=None)
        df_cleaned = pd.read_csv(table_path, nrows=None)
        df_raw = pd.read_csv(raw_path, nrows=None)

    df_cleaned = df_cleaned.fillna('')
    df_cleaned_char = df_cleaned_char.fillna('')

    print(df_cleaned.columns)
    print(df_cleaned_char.columns)

    print(df_cleaned.shape)
    print(df_cleaned_char.shape)

    vocabs_path = os.path.join(args.data_dir, 'vocabs')
    uni2idx_path = os.path.join(vocabs_path, 'uni2idx.pkl')
    bi2idx_path = os.path.join(vocabs_path, 'bi2idx.pkl')
    char2idx_path = os.path.join(vocabs_path, 'char2idx.pkl')

    all_files_exist = os.path.isfile(uni2idx_path) and os.path.isfile(bi2idx_path) and os.path.isfile(char2idx_path)

    assert all_files_exist, 'Dont have vocab files'

    print('Loading vocabs file from {}'.format(vocabs_path))
    uni2idx = pickle.load(open(uni2idx_path, 'rb'))
    bi2idx = pickle.load(open(bi2idx_path, 'rb'))
    char2idx = pickle.load(open(char2idx_path, 'rb'))

    print('uni vocab len = {}'.format(len(uni2idx)))
    print('bi vocab len = {}'.format(len(bi2idx)))
    print('char vocab len = {}'.format(len(char2idx)))

    vectorizing_params = {
        'uni2idx': uni2idx,
        'bi2idx': bi2idx,
        'char2idx': char2idx,
        'seq_words_maxlen': 20,
        'seq_bis_maxlen': 20,
        'seq_chars_maxlen': 100
    }

    def vect_char(x): return vectorize_chars(x, params=vectorizing_params)
    def vect_wb(x): return vectorize_uni_bi(x, params=vectorizing_params)

    def vect_char_(x): return vectorize_chars(x, params=vectorizing_params, trunc='pre')
    def vect_wb_(x): return vectorize_uni_bi(x, params=vectorizing_params, trunc='pre')

    with Pool(15) as p:
        print('1')
        wb_res = p.map(vect_wb_, df_cleaned['context'])
        c_res = np.zeros((len(wb_res), 100), int) # p.map(vect_char_, df_cleaned_char['context'])
    with Pool(15) as p:
        print('2')
        #c_res1 = c_res # p.map(vect_char, df_cleaned_char['question'])
        wb_res1 = p.map(vect_wb, df_cleaned['question'])
    with Pool(15) as p:
        print('3')
        #c_res2 = c_res # p.map(vect_char, df_cleaned_char['reply'])
        wb_res2 = p.map(vect_wb, df_cleaned['reply'])
    with Pool(15) as p:
        print('4')
        #c_res3 = c_res # p.map(vect_char, df_cleaned_char['fact1'])
        wb_res3 = p.map(vect_wb, df_cleaned['fact1'])
    with Pool(15) as p:
        print('5')
        #c_res4 = c_res # p.map(vect_char, df_cleaned_char['fact2'])
        wb_res4 = p.map(vect_wb, df_cleaned['fact2'])
    with Pool(15) as p:
        print('6')
        #c_res5 = c_res # p.map(vect_char, df_cleaned_char['fact3'])
        wb_res5 = p.map(vect_wb, df_cleaned['fact3'])
    with Pool(15) as p:
        print('7')
        #c_res6 = c_res # p.map(vect_char, df_cleaned_char['fact4'])
        wb_res6 = p.map(vect_wb, df_cleaned['fact4'])
    with Pool(15) as p:
        print('8')
        #c_res7 = c_res # p.map(vect_char, df_cleaned_char['fact5'])
        wb_res7 = p.map(vect_wb, df_cleaned['fact5'])

    print('saving...')
    data = np.hstack((wb_res, c_res,
                      wb_res1, c_res,
                      wb_res2, c_res,
                      wb_res3, c_res,
                      wb_res4, c_res,
                      wb_res5, c_res,
                      wb_res6, c_res,
                      wb_res7, c_res)).reshape(-1, 8, 140)

    responses = np.hstack((wb_res2, c_res)).reshape(-1, 140)
    unique_responses, indexes = np.unique(responses, axis=0, return_index=True)
    unique_data = data[indexes]

    raw_responses = df_raw['reply'].values
    raw_responses = raw_responses[indexes]

    print('data shape = {}'.format(data.shape))
    print('responses shape = {}'.format(unique_responses.shape))
    print('raw responses num = {}'.format(len(raw_responses)))

    Y = df_cleaned['labels'].values.ravel()
    unique_labels = Y[indexes]

    Ytr, Yev, Xtr, Xev = train_test_split(Y,
                                          data,
                                          test_size=0.1,
                                          random_state=24)

    pickle.dump(unique_responses, open(os.path.join(args.data_dir, 'responses.pkl'), 'wb'))
    pickle.dump(raw_responses, open(os.path.join(args.data_dir, 'raw_responses.pkl'), 'wb'))

    sample_path = os.path.join(args.data_dir, 'sample')

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    print('Converting to TFRecords...')
    convert_to_records([data, Y], 'full', args.data_dir)
    convert_to_records([Xtr, Ytr], 'train', args.data_dir)
    convert_to_records([Xev, Yev], 'eval', args.data_dir)
    convert_to_records([unique_data, unique_labels], 'unique_data', args.data_dir)

    convert_to_records([data[:1000], Y[:1000]], 'full', sample_path)
    convert_to_records([Xtr[:1000], Ytr[:1000]], 'train', sample_path)
    convert_to_records([Xev[:1000], Yev[:1000]], 'eval', sample_path)
