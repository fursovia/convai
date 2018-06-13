"""tokenize and vectorize"""

import pickle
import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk import ngrams
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re


parser = argparse.ArgumentParser()
parser.add_argument('--features', default='N', help="Whether to do some feature engineering")
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--vocab_size', type=int, default=10000)

def clean(text):
    text = text.lower()
    # оставляем только буквы
    text = re.sub("[^a-z' ]+", '', text)
    return text


def vectorize_text(text, word2idx, maxlen=40):
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

    return pad_sequences([vec_seen], maxlen=maxlen, truncating='post')[0]


if __name__ == '__main__':

    args = parser.parse_args()

    train_raw_data_path = os.path.join(args.data_dir, 'initial/train_both_original_no_cands.txt')
    valid_raw_data_path = os.path.join(args.data_dir, 'initial/valid_both_original_no_cands.txt')

    with open(train_raw_data_path, 'r', encoding='utf-8') as file:
        train_raw_data = file.readlines()

    with open(valid_raw_data_path, 'r', encoding='utf-8') as file:
        valid_raw_data = file.readlines()

    raw_data = train_raw_data + valid_raw_data

    pers_info_attrs = ['your persona:', 'partner\'s persona:']

    # находим все диалоги
    dialogs = []
    personal_infos = []
    all_utterances = []

    for i, line in tqdm(enumerate(raw_data)):
        if line[:2] == '1 ':  # line starts with
            if i != 0:
                dialogs.append(dialog)
                infos = [first_persona, second_persona]
                personal_infos.append(infos)

            first_persona = []
            second_persona = []
            dialog = []

        if pers_info_attrs[0] in line:
            attr = ' '.join(line.split(':')[1:])
            first_persona.append(attr)
        elif pers_info_attrs[1] in line:
            attr = ' '.join(line.split(':')[1:])
            second_persona.append(attr)
        else:
            quest_reply = line[2:].split('\t')
            all_utterances.extend(quest_reply)
            dialog.extend(quest_reply)

    all_utterances_cleaned = []
    for utt in all_utterances:
        all_utterances_cleaned.append(clean(utt))
    all_utterances_cleaned = list(set(all_utterances_cleaned))

    all_infos_cleaned = []
    info_lens = []
    for infos in personal_infos:
        len1, len2 = list(map(len, infos))
        info_lens.append(len1)
        info_lens.append(len2)
        for j in infos:
            for i in j:
                all_infos_cleaned.append(clean(i))
    all_infos_cleaned = list(set(all_infos_cleaned))

    print('Mean info len = {}'.format(np.mean(info_lens)))
    print('Median info len = {}'.format(np.median(info_lens)))
    print('Max info len = {}'.format(np.max(info_lens)))
    print('Min info len = {}'.format(np.min(info_lens)))

    # чистим диалоги и персональную инфу
    cleaned_dials = []
    cleaned_infos = []

    for i, dialog in enumerate(tqdm(dialogs)):
        cleaned_dial = []

        first_cleaned = []
        second_cleaned = []
        first_info, second_info = personal_infos[i]
        for info in first_info:
            cleaned_info = clean(info)
            first_cleaned.append(cleaned_info)
        for info in second_info:
            cleaned_info = clean(info)
            second_cleaned.append(cleaned_info)
        cleaned_infos.append([first_cleaned, second_cleaned])

        for idx, mes in enumerate(dialog):
            cleaned_mes = clean(mes)
            cleaned_dial.append(cleaned_mes)
        cleaned_dials.append(cleaned_dial)

    # диалог начинает другой человек, а мы отвечаем
    X = []

    for i, dialog in enumerate(tqdm(cleaned_dials)):

        for idx, mes in enumerate(dialog):
            if idx % 2 == 0:
                personal_info = cleaned_infos[i][1]
            else:
                personal_info = cleaned_infos[i][0]

            if idx > 1:
                context = ' '.join(dialog[:(idx - 1)])
                question = dialog[idx - 1]
                reply = dialog[idx]
                x_small = [context, question, reply, personal_info]
                X.append(x_small)
            else:
                continue

    print('Num of positive examples = {}'.format(len(X)))

    # получаем отрицательные примеры

    np.random.shuffle(all_utterances_cleaned)

    X_neg = X
    for i in range(len(X_neg)):
        X_neg[i][2] = clean(all_utterances_cleaned[i])  # рандомный ответ

    print('Num of negative examples = {}'.format(len(X_neg)))


    # TODO: придумать фичи
    if args.features == 'Y':
        pass

    # получаем словарь
    num_words = int(args.vocab_size / 2)
    num_bigrams = int(args.vocab_size / 2)

    corpus = all_infos_cleaned + all_utterances_cleaned
    words = ' '.join(corpus).split()
    bigrams = ngrams(words, 2)

    uni_counter = Counter(words)
    bi_counter = Counter(bigrams)

    print('Num of unigrams: {0}'.format(len(uni_counter.most_common())))
    print('Num of bigrams: {0}'.format(len(bi_counter.most_common())))

    uni2idx = {word[0]: i + 1 for i, word in enumerate(uni_counter.most_common(num_words))}
    bi2idx = {word[0]: num_words + 1 + i for i, word in enumerate(bi_counter.most_common(num_bigrams))}

    word2idx = {}
    word2idx.update(uni2idx)
    word2idx.update(bi2idx)

    word2idx_path = os.path.join(args.data_dir, 'word2idx.pkl')
    pickle.dump(word2idx, open(word2idx_path, 'wb'))
    print('word2idx file saved at {}'.format(word2idx_path))

    # векторизуем текст
    Y = []
    context_vect = []
    question_vect = []
    reply_vect = []
    info_vect = []

    for i, dial in enumerate(tqdm(X)):
        cont = vectorize_text(dial[0], word2idx, 80)
        ques = vectorize_text(dial[1], word2idx)
        reply = vectorize_text(dial[2], word2idx)
        neg_reply = vectorize_text(X_neg[i][2], word2idx)

        context_vect.append(cont)
        question_vect.append(ques)
        reply_vect.append(reply)
        Y.append(1)
        reply_vect.append(neg_reply)
        Y.append(0)

        info_ = dial[3]
        for j in range(5):  # 5 фактов о каждом
            try:
                vect_info = vectorize_text(info_[j], word2idx)
                info_vect.append(vect_info)
            except IndexError:
                info_vect.append(np.zeros_like(ques))  # длина вопроса равна длине факта

    context_vect = context_vect + context_vect
    question_vect = question_vect + question_vect
    info_vect = info_vect + info_vect

    context_vect = np.array(context_vect, int)
    question_vect = np.array(question_vect, int)
    reply_vect = np.array(reply_vect, int)
    info_vect = np.array(info_vect).reshape(-1, 200)
    Y = np.array(Y, int)

    print('Context shape = {}'.format(context_vect.shape))
    print('Question shape = {}'.format(question_vect.shape))
    print('Reply shape = {}'.format(reply_vect.shape))
    print('Personal info shape = {}'.format(info_vect.shape))

    # сохраняем данные
    Ytr, Yev, Ctr, Cev, Qtr, Qev, Rtr, Rev, Itr, Iev = train_test_split(Y,
                                                                        context_vect,
                                                                        question_vect,
                                                                        reply_vect,
                                                                        info_vect,
                                                                        test_size=0.1,
                                                                        random_state=24)

    data_path = args.data_dir
    train_path = os.path.join(data_path, 'train')
    valid_path = os.path.join(data_path, 'eval')
    sample_train_path = os.path.join(data_path, 'sample/train')
    sample_valid_path = os.path.join(data_path, 'sample/eval')

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    if not os.path.exists(sample_train_path):
        os.makedirs(sample_train_path)

    if not os.path.exists(sample_valid_path):
        os.makedirs(sample_valid_path)

    pickle.dump(Ytr, open(os.path.join(train_path, 'Y.pkl'), 'wb'))
    pickle.dump(Yev, open(os.path.join(valid_path, 'Y.pkl'), 'wb'))
    pickle.dump(Ctr, open(os.path.join(train_path, 'C.pkl'), 'wb'))
    pickle.dump(Cev, open(os.path.join(valid_path, 'C.pkl'), 'wb'))
    pickle.dump(Qtr, open(os.path.join(train_path, 'Q.pkl'), 'wb'))
    pickle.dump(Qev, open(os.path.join(valid_path, 'Q.pkl'), 'wb'))
    pickle.dump(Rtr, open(os.path.join(train_path, 'R.pkl'), 'wb'))
    pickle.dump(Rev, open(os.path.join(valid_path, 'R.pkl'), 'wb'))
    pickle.dump(Itr, open(os.path.join(train_path, 'I.pkl'), 'wb'))
    pickle.dump(Iev, open(os.path.join(valid_path, 'I.pkl'), 'wb'))

    pickle.dump(Ytr[:100], open(os.path.join(sample_train_path, 'Y.pkl'), 'wb'))
    pickle.dump(Yev[:100], open(os.path.join(sample_valid_path, 'Y.pkl'), 'wb'))
    pickle.dump(Ctr[:100], open(os.path.join(sample_train_path, 'C.pkl'), 'wb'))
    pickle.dump(Cev[:100], open(os.path.join(sample_valid_path, 'C.pkl'), 'wb'))
    pickle.dump(Qtr[:100], open(os.path.join(sample_train_path, 'Q.pkl'), 'wb'))
    pickle.dump(Qev[:100], open(os.path.join(sample_valid_path, 'Q.pkl'), 'wb'))
    pickle.dump(Rtr[:100], open(os.path.join(sample_train_path, 'R.pkl'), 'wb'))
    pickle.dump(Rev[:100], open(os.path.join(sample_valid_path, 'R.pkl'), 'wb'))
    pickle.dump(Itr[:100], open(os.path.join(sample_train_path, 'I.pkl'), 'wb'))
    pickle.dump(Iev[:100], open(os.path.join(sample_valid_path, 'I.pkl'), 'wb'))

    pickle.dump(Y, open(os.path.join(train_path, 'Y.pkl'), 'wb'))
    pickle.dump(context_vect, open(os.path.join(train_path, 'C.pkl'), 'wb'))
    pickle.dump(question_vect, open(os.path.join(train_path, 'Q.pkl'), 'wb'))
    pickle.dump(reply_vect, open(os.path.join(train_path, 'R.pkl'), 'wb'))
    pickle.dump(info_vect, open(os.path.join(train_path, 'I.pkl'), 'wb'))

    print('Data saved at {}'.format(train_path))
    print('and at {}'.format(valid_path))
