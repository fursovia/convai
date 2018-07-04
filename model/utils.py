"""Вспомогательные функции"""

import json
import pandas as pd
import os
import numpy as np
import pickle
import tensorflow as tf
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import ngrams
from keras.preprocessing.sequence import pad_sequences
from multiprocessing import Pool


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


def get_coefs(arr):
    word, arr = arr[:-300], arr[-300:]
    return ' '.join(word), np.array(arr, dtype=np.float64)


def get_embeddings(params):
    word2idx_file = os.path.join(params.data_path, 'word2idx.pkl')
    fasttext_file = os.path.join(params.data_path, 'fasttext.vec')

    word2idx = pickle.load(open(word2idx_file, 'rb'))
    embeddings_index = dict(get_coefs(o.strip().split()) for o in open(fasttext_file, encoding='utf-8'))

    embedding_matrix = np.zeros(((params.vocab_size + 1), params.embedding_size))

    for word, i in word2idx.items():
        if type(word) == tuple:
            word = ' '.join(word)

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(-0.1, 0.1, params.embedding_size)

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
            cont = X[index][0]
            quest = X[index][1]
            resp = X[index][2]
            facts = X[index][3:].ravel()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': _int64_feature([Y_]),
                        'cont': _int64_feature(cont),
                        'quest': _int64_feature(quest),
                        'resp': _int64_feature(resp),
                        'facts': _int64_feature(facts)
                    }
                )
            )
            writer.write(example.SerializeToString())


def decode(serialized_example):
    """Parses an image and label from the given `serialized_example`."""

    features_pattern = {'label': tf.FixedLenFeature([], tf.int64)}

    features_pattern['cont'] = tf.FixedLenFeature([140], tf.int64)
    features_pattern['quest'] = tf.FixedLenFeature([140], tf.int64)
    features_pattern['resp'] = tf.FixedLenFeature([140], tf.int64)
    features_pattern['facts'] = tf.FixedLenFeature([140*5], tf.int64)
    features = tf.parse_single_example(serialized_example, features=features_pattern)
    label = features.pop('label')
    sent = features

    return sent, label


def clean(text, stem=True):
    text = text.strip().lower()
    text = re.sub('[^\w\s]', ' ', text)
    text = re.sub('^\d+\s|\s\d+\s|\s\d+$', " <num> ", text)
    if stem:
        return ' '.join(snowball_stemmer.stem(word) for word in text.split())
    else:
        return text


def vectorize_text(text, word2idx, maxlen=20, truncating_type='post', only_words=True):
    vec_seen = []

    words_ = text.split()

    for word in words_:
        try:
            vec_seen.append(word2idx[word])
        except:
            continue

    bi_ = ngrams(words_, 2)
    for bi in bi_:
        try:
            vec_seen.append(word2idx[bi])
        except:
            continue

    return pad_sequences([vec_seen], maxlen=maxlen, truncating=truncating_type)[0]


def text2vec(dict_from_parlai, word2idx):
    personal_info = []  # нужно иметь 5 фактов
    dial = []
    raw_dial = []
    cands = dict_from_parlai['label_candidates']
    splitted_text = dict_from_parlai['text'].split('\n')
    true_ans = dict_from_parlai['eval_labels'][0]
    for i, cand in enumerate(cands):
        if cand == true_ans:
            true_answer_id = i

    cleaned_cands = []
    for mes in cands:
        cleaned_cands.append(clean(mes))

    for mes in splitted_text:
        if 'your persona:' in mes:
            personal_info.append(clean(' '.join(mes.split(':')[1:])))
        else:
            dial.append(clean(mes))
            raw_dial.append(mes)

    dial_len = len(dial)

    if dial_len == 1:
        cont = ''
        quest = dial[0]

    if dial_len == 2:
        cont = dial[0]
        quest = dial[1]

    if dial_len == 3:
        cont = ' '.join(dial[0:2])
        quest = dial[2]

    if dial_len > 3:
        cont = ' '.join(dial[dial_len-4:dial_len-1])  # контекст длины 3
        quest = dial[dial_len-1]

    info_5 = []
    for i in range(5):
        try:
            info_5.append(personal_info[i])
        except IndexError:
            info_5.append('')

    X = []
    for cand in cleaned_cands:
        X.append([cont, quest, cand, info_5])

    context_vect = []
    question_vect = []
    reply_vect = []
    info_vect = []
    for i, dial in enumerate(X):
        cont = vectorize_text(dial[0], word2idx, 60, 'pre')
        ques = vectorize_text(dial[1], word2idx)
        reply = vectorize_text(dial[2], word2idx)

        context_vect.append(cont)
        question_vect.append(ques)
        reply_vect.append(reply)

        info_ = dial[3]
        for j in range(5):  # 5 фактов о каждом
            vect_info = vectorize_text(info_[j], word2idx)
            info_vect.append(vect_info)

    data = np.hstack((context_vect, question_vect, reply_vect, np.array(info_vect).reshape(-1, 100)))

    return data, true_answer_id, cands[true_answer_id], raw_dial, cands


def vectorize_chars(text, params, trunc='post'):
    chars_seen = []
    chars_ = ngrams(text, 3)
    for char in chars_:
        try:
            chars_seen.append(params['char2idx'][char])
        except KeyError:
            chars_seen.append(params['char2idx']['u_k'])
    c_vect = pad_sequences([chars_seen], maxlen=params['seq_chars_maxlen'], truncating=trunc)[0]
    return list(c_vect)


def vectorize_uni_bi(text, params, trunc='post'):
    words_seen = []
    bis_seen = []
    words_ = text.split()
    bi_ = ngrams(words_, 2)
    for word in words_:
        try:
            words_seen.append(params['uni2idx'][word])
        except KeyError:
            words_seen.append(params['uni2idx']['u_k'])

    for bi in bi_:
        try:
            bis_seen.append(params['bi2idx'][bi])
        except KeyError:
            bis_seen.append(params['bi2idx']['u_k'])

    w_vect = pad_sequences([words_seen], maxlen=params['seq_words_maxlen'], truncating=trunc)[0]
    b_vect = pad_sequences([bis_seen], maxlen=params['seq_bis_maxlen'], truncating=trunc)[0]
    return list(w_vect) + list(b_vect)


def clean2(text):
    return clean(text, stem=True)


# def inference_time(dict_from_tg, responses):
#     cont = clean2(dict_from_tg['context'])
#     quest = clean2(dict_from_tg['question'])
#     resp = responses
#     facts = list(map(clean2, dict_from_tg['facts']))
#
#     conts = [cont] * len(resp)
#     quests = [quest] * len(resp)
#     f1 = [facts[0]] * len(resp)
#     f2 = [facts[1]] * len(resp)
#     f3 = [facts[2]] * len(resp)
#     f4 = [facts[3]] * len(resp)
#     f5 = [facts[4]] * len(resp)
#
#     df = pd.DataFrame({'context': conts,
#                        'question': quests,
#                        'reply': resp,
#                        'fact1': f1,
#                        'fact2': f2,
#                        'fact3': f3,
#                        'fact4': f4,
#                        'fact5': f5})
#     return df


def inference_time(dict_from_tg, responses, vocabs, repeat=None):

    uni2idx, bi2idx, char2idx = vocabs

    vectorizing_params = {
        'uni2idx': uni2idx,
        'bi2idx': bi2idx,
        'char2idx': char2idx,
        'seq_words_maxlen': 20,
        'seq_bis_maxlen': 20,
        'seq_chars_maxlen': 100
    }

    def vect_char(x): return np.array(vectorize_chars(x, params=vectorizing_params), int).reshape(-1, 100)
    def vect_wb(x): return np.array(vectorize_uni_bi(x, params=vectorizing_params), int).reshape(-1, 40)

    def vect_char_(x): return np.array(vectorize_chars(x, params=vectorizing_params, trunc='pre'), int).reshape(-1, 100)
    def vect_wb_(x): return np.array(vectorize_uni_bi(x, params=vectorizing_params, trunc='pre'), int).reshape(-1, 40)

    cont = clean2(' '.join(dict_from_tg['context']))
    quest = clean2(dict_from_tg['question'])
    resp = responses
    facts = list(map(clean2, dict_from_tg['facts']))

    f1 = facts[0]
    f2 = facts[1]
    f3 = facts[2]
    f4 = facts[3]
    f5 = facts[4]

    if repeat is None:
        rep = resp.shape[0]
        wb_res2 = resp[:, :40]
        c_res2 = resp[:, 40:140]
    else:
        rep = 1
        wb_res2 = resp[0, :40].reshape(-1, 40)
        c_res2 = resp[0, 40:140].reshape(-1, 100)

    wb_res = np.repeat(vect_wb_(cont), rep, axis=0)
    c_res = np.repeat(vect_char_(cont), rep, axis=0)
    wb_res1 = np.repeat(vect_wb(quest), rep, axis=0)
    c_res1 = np.repeat(vect_char(quest), rep, axis=0)
    wb_res3 = np.repeat(vect_wb(f1), rep, axis=0)
    c_res3 = np.repeat(vect_char(f1), rep, axis=0)
    wb_res4 = np.repeat(vect_wb(f2), rep, axis=0)
    c_res4 = np.repeat(vect_char(f2), rep, axis=0)
    wb_res5 = np.repeat(vect_wb(f3), rep, axis=0)
    c_res5 = np.repeat(vect_char(f3), rep, axis=0)
    wb_res6 = np.repeat(vect_wb(f4), rep, axis=0)
    c_res6 = np.repeat(vect_char(f4), rep, axis=0)
    wb_res7 = np.repeat(vect_wb(f5), rep, axis=0)
    c_res7 = np.repeat(vect_char(f5), rep, axis=0)

    data = np.hstack((wb_res, c_res,
                      wb_res1, c_res1,
                      wb_res2, c_res2,
                      wb_res3, c_res3,
                      wb_res4, c_res4,
                      wb_res5, c_res5,
                      wb_res6, c_res6,
                      wb_res7, c_res7)).reshape(-1, 8, 140)

    dict_to_return = {}
    dict_to_return['cont'] = data[:,0]
    dict_to_return['quest'] = data[:,1]
    dict_to_return['resp'] = data[:,2]
    dict_to_return['facts'] = data[:,2:]

    return dict_to_return

