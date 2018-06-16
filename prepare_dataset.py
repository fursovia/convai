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
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


parser = argparse.ArgumentParser()
parser.add_argument('--features', default='N', help="Whether to do some feature engineering")
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--vocab_size', type=int, default=16000)

snowball_stemmer = SnowballStemmer("english")
stop_words = stopwords.words("english")


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


if __name__ == '__main__':

    args = parser.parse_args()

    train_raw_data_path1 = os.path.join(args.data_dir, 'initial/train_both_original.txt')
    valid_raw_data_path1 = os.path.join(args.data_dir, 'initial/valid_both_original.txt')

    train_raw_data_path2 = os.path.join(args.data_dir, 'initial/train_both_revised.txt')
    valid_raw_data_path2 = os.path.join(args.data_dir, 'initial/valid_both_revised.txt')

    with open(train_raw_data_path1, 'r', encoding='utf-8') as file:
        train_raw_data1 = file.readlines()

    with open(valid_raw_data_path1, 'r', encoding='utf-8') as file:
        valid_raw_data1 = file.readlines()

    with open(train_raw_data_path2, 'r', encoding='utf-8') as file:
        train_raw_data2 = file.readlines()

    with open(valid_raw_data_path2, 'r', encoding='utf-8') as file:
        valid_raw_data2 = file.readlines()

    train_raw_data = train_raw_data1 + train_raw_data2
    valid_raw_data = valid_raw_data1 + valid_raw_data2

    raw_data = train_raw_data + valid_raw_data

    path1 = os.path.join(args.data_dir, 'dials.pkl')
    path2 = os.path.join(args.data_dir, 'infos.pkl')
    path3 = os.path.join(args.data_dir, 'cands.pkl')

    if os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3):
        cleaned_dials = pickle.load(open(path1, 'rb'))
        cleaned_infos = pickle.load(open(path2, 'rb'))
        cleaned_cands = pickle.load(open(path3, 'rb'))

    else:
        # находим все диалоги
        dialogs = []
        personal_infos = []
        candidates = []
        all_utterances = []

        for i, line in tqdm(enumerate(raw_data)):
            if line.startswith('1 '):
                if i != 0:
                    dialogs.append(dialog)
                    infos = [first_persona, second_persona]
                    personal_infos.append(infos)
                    candidates.append(candidates_)

                first_persona = []
                second_persona = []
                dialog = []
                candidates_ = []

            if 'your persona:' in line:
                attr = ' '.join(line.split(':')[1:])
                first_persona.append(attr)
            elif 'partner\'s persona:' in line:
                attr = ' '.join(line.split(':')[1:])
                second_persona.append(attr)
            else:
                splitted_line = line[2:].split('\t')
                quest_reply = splitted_line[:2]
                cands = splitted_line[3].split('|')

                dialog.extend(quest_reply)
                all_utterances.extend(quest_reply)
                candidates_.append(cands)

        print('num of dialogs = {}'.format(len(dialogs)))
        print('num of personal infos = {}'.format(len(personal_infos)))
        print('num of candidates = {}'.format(len(candidates)))

        print('cleaning utterances...')
        all_utterances_cleaned = []
        for utt in tqdm(all_utterances):
            all_utterances_cleaned.append(clean(utt))

        # чистим диалоги и персональную инфу
        cleaned_dials = []
        cleaned_infos = []
        cleaned_cands = []

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

            curr_cands = candidates[i]
            cleaned_cands_ = []
            for cand in curr_cands:
                cleaned_mes = []
                for mes in cand:
                    cleaned_mes.append(clean(mes))
                cleaned_cands_.append(cleaned_mes)
            cleaned_cands.append(cleaned_cands_)

        print('num of dialogs = {}'.format(len(cleaned_dials)))
        print('num of personal infos = {}'.format(len(cleaned_infos)))
        print('num of candidates = {}'.format(len(cleaned_cands)))

        pickle.dump(cleaned_dials, open(os.path.join(args.data_dir, 'dials.pkl'), 'wb'))
        pickle.dump(cleaned_infos, open(os.path.join(args.data_dir, 'infos.pkl'), 'wb'))
        pickle.dump(cleaned_cands, open(os.path.join(args.data_dir, 'cands.pkl'), 'wb'))

    # диалог начинает другой человек, а мы отвечаем
    X = []
    Y = []

    for i, dialog in enumerate(tqdm(cleaned_dials)):
        for idx, mes in enumerate(dialog):
            if idx % 2 == 0:
                personal_info = cleaned_infos[i][1]
            else:
                personal_info = cleaned_infos[i][0]

            if idx == 1:
                context = ''
            if idx == 2:
                context_len = 1
                context = ' '.join(dialog[(idx - 1 - context_len):(idx - 1)])
            if idx == 3:
                context_len = 2
                context = ' '.join(dialog[(idx - 1 - context_len):(idx - 1)])
            if idx > 3:
                context_len = 3
                context = ' '.join(dialog[(idx - 1 - context_len):(idx - 1)])

            if idx >= 1:
                question = dialog[idx - 1]
                reply = dialog[idx]
                x_small = [context, question, reply, personal_info]
                X.append(x_small)
                Y.append(1)

                if idx % 2 == 1:
                    idxx = int(idx/2)
                    neg_cands = cleaned_cands[i]
                    curr_negs = neg_cands[idxx]  # 20 кандидатов
                    for cand in curr_negs:
                        x_small = [context, question, cand, personal_info]
                        X.append(x_small)
                        Y.append(0)
                # VERY SLOW
                # else:
                #     curr_negs = np.random.choice(all_utterances_cleaned, 20, replace=False)
                #     for cand in curr_negs:
                #         x_small = [context, question, cand, personal_info]
                #         X.append(x_small)
                #         Y.append(0)

    # TODO: придумать фичи
    if args.features == 'Y':
        pass

    corpus = []
    for i in range(len(cleaned_dials)):
        dial = cleaned_dials[i]
        infos = cleaned_infos[i]
        cands_ = cleaned_cands[i]
        for mes in dial:
            corpus.append(mes)
        for inf_ in infos:
            for mes in inf_:
                corpus.append(mes)
        for c in cands_:
            for mes in c:
                corpus.append(mes)

    corpus = list(set(corpus))

    # получаем словарь
    word2idx_path = os.path.join(args.data_dir, 'word2idx.pkl')
    if os.path.exists(word2idx_path):
        print('Loading word2idx file from {}'.format(word2idx_path))
        word2idx = pickle.load(open(word2idx_path, 'rb'))
    else:
        num_words = int(args.vocab_size / 2)
        num_bigrams = int(args.vocab_size / 2)

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

        pickle.dump(word2idx, open(word2idx_path, 'wb'))
        pickle.dump(corpus, open(os.path.join(args.data_dir, 'corpus.pkl'), 'wb'))
        print('word2idx file saved at {}'.format(word2idx_path))
        print('corpus file saved at {}'.format(args.data_dir))

    # векторизуем текст
    context_vect = []
    question_vect = []
    reply_vect = []
    info_vect = []

    for i, dial in enumerate(tqdm(X)):
        cont = vectorize_text(dial[0], word2idx, 60, 'pre')
        ques = vectorize_text(dial[1], word2idx)
        reply = vectorize_text(dial[2], word2idx)

        context_vect.append(cont)
        question_vect.append(ques)
        reply_vect.append(reply)

        info_ = dial[3]
        for j in range(5):  # 5 фактов о каждом
            try:
                vect_info = vectorize_text(info_[j], word2idx)
                info_vect.append(vect_info)
            except IndexError:
                info_vect.append(np.zeros_like(ques))  # длина вопроса равна длине факта

    context_vect = np.array(context_vect, int).reshape(-1, 60)
    question_vect = np.array(question_vect, int).reshape(-1, 20)
    reply_vect = np.array(reply_vect, int).reshape(-1, 20)
    info_vect = np.array(info_vect).reshape(-1, 100)
    Y = np.array(Y, int).reshape(-1, 1)

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
    print('...and at {}'.format(valid_path))
