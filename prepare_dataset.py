"""скрипт сохраняет три словаря и две таблицы"""

import pickle
import os
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk import ngrams
from model.utils import clean
import pandas as pd
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--num_uni', type=int, default=20000)
parser.add_argument('--num_bi', type=int, default=20000)
parser.add_argument('--num_chars', type=int, default=10000)
parser.add_argument('--min_freq', type=int, default=5)
parser.add_argument('--neg_to_pos', type=int, default=5,
                    help='How many negative examples to one positive. NO MORE THAN 20')


if __name__ == '__main__':

    args = parser.parse_args()

    train_raw_data_path1 = os.path.join(args.data_dir, 'initial/train_both_original.txt')
    valid_raw_data_path1 = os.path.join(args.data_dir, 'initial/valid_both_original.txt')

    train_raw_data_path2 = os.path.join(args.data_dir, 'initial/train_both_revised.txt')
    valid_raw_data_path2 = os.path.join(args.data_dir, 'initial/valid_both_revised.txt')

    a = pd.read_csv('/data/i.fursov/mityai/clean_only_ctx_df.csv')
    b = pd.read_csv('/data/i.fursov/mityai/clean2_ctx_df.csv')
    c = pd.read_csv('/data/i.fursov/mityai/raw_sub_ctx_df.csv')

    a = a.fillna('')
    b = b.fillna('')
    c = c.fillna('')
    mit_clean = a[a['labels'] != 0]
    mit_clean_chars = b[b['labels'] != 0]
    mit_raw = c[c['labels'] != 0]

    print(mit_clean.shape)
    print(mit_clean_chars.shape)
    print(mit_raw.shape)

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

    print('preparing data for further handling...')
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

    print('getting X and Y...')
    # диалог начинает другой человек, а мы отвечаем
    X = []
    Y = []

    for i, dialog in enumerate(tqdm(dialogs)):
        for idx, mes in enumerate(dialog):
            if idx % 2 == 0:
                personal_info = personal_infos[i][1]
            else:
                personal_info = personal_infos[i][0]

            personal_info_ = []
            for inf_id in range(5):
                try:
                    personal_info_.append(personal_info[inf_id])
                except IndexError:
                    personal_info_.append('')

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
                x_small = [context, question, reply]
                x_small.extend(personal_info_)
                X.append(x_small)
                Y.append(1)

                if idx % 2 == 1:
                    idxx = int(idx / 2)
                    neg_cands = candidates[i]
                    curr_negs = neg_cands[idxx]  # 20 кандидатов
                    for neg_i in range(args.neg_to_pos):
                        cand = curr_negs[neg_i]
                        x_small = [context, question, cand]
                        x_small.extend(personal_info_)
                        X.append(x_small)
                        Y.append(0)

    columns = ['context', 'question', 'reply', 'fact1', 'fact2', 'fact3', 'fact4', 'fact5', 'labels']
    XY = np.hstack((np.array(X, object), np.array(Y, int).reshape(-1, 1)))
    df = pd.DataFrame(XY, columns=columns)
    print(df.shape)

    #df.to_csv(os.path.join(args.data_dir, 'raw_df.csv'), index=False)

    print('cleaning...')
    with Pool(50) as p:
        print('1')
        context_cleaned = p.map(clean, df['context'])
        print('2')
        questions_cleaned = p.map(clean, df['question'])
        print('3')
        replies_cleaned = p.map(clean, df['reply'])
        print('4')
        fact1_cleaned = p.map(clean, df['fact1'])
        print('5')
        fact2_cleaned = p.map(clean, df['fact2'])
        print('6')
        fact3_cleaned = p.map(clean, df['fact3'])
        print('7')
        fact4_cleaned = p.map(clean, df['fact4'])
        print('8')
        fact5_cleaned = p.map(clean, df['fact5'])

    df_cleaned = pd.DataFrame({
        'context': context_cleaned,
        'question': questions_cleaned,
        'reply': replies_cleaned,
        'fact1': fact1_cleaned,
        'fact2': fact2_cleaned,
        'fact3': fact3_cleaned,
        'fact4': fact4_cleaned,
        'fact5': fact5_cleaned,
        'labels': df['labels'].values
    })

    print(df_cleaned.shape)

    #df_cleaned.to_csv(os.path.join(args.data_dir, 'cleaned_df.csv'), index=False)

    def clean2(text):
        return clean(text, stem=False)

    print('cleaning for the second time (for chars)...')
    with Pool(50) as p:
        print('1')
        context_cleaned = p.map(clean2, df['context'])
        print('2')
        questions_cleaned = p.map(clean2, df['question'])
        print('3')
        replies_cleaned = p.map(clean2, df['reply'])
        print('4')
        fact1_cleaned = p.map(clean2, df['fact1'])
        print('5')
        fact2_cleaned = p.map(clean2, df['fact2'])
        print('6')
        fact3_cleaned = p.map(clean2, df['fact3'])
        print('7')
        fact4_cleaned = p.map(clean2, df['fact4'])
        print('8')
        fact5_cleaned = p.map(clean2, df['fact5'])

    df_cleaned_char = pd.DataFrame({
        'context': context_cleaned,
        'question': questions_cleaned,
        'reply': replies_cleaned,
        'fact1': fact1_cleaned,
        'fact2': fact2_cleaned,
        'fact3': fact3_cleaned,
        'fact4': fact4_cleaned,
        'fact5': fact5_cleaned,
        'labels': df['labels'].values
    })

    print(df_cleaned_char.shape)

    #df_cleaned_char.to_csv(os.path.join(args.data_dir, 'cleaned_char_df.csv'), index=False)

    # получаем словарь
    vocabs_path = os.path.join(args.data_dir, 'vocabs')
    uni2idx_path = os.path.join(vocabs_path, 'uni2idx.pkl')
    bi2idx_path = os.path.join(vocabs_path, 'bi2idx.pkl')
    char2idx_path = os.path.join(vocabs_path, 'char2idx.pkl')

    if not os.path.exists(vocabs_path):
        os.makedirs(vocabs_path)

    df = df[columns].append(mit_raw[columns])
    df_cleaned = df_cleaned[columns].append(mit_clean[columns])
    df_cleaned_char = df_cleaned_char[columns].append(mit_clean_chars[columns])

    print(df.shape)
    print(df_cleaned.shape)
    print(df_cleaned_char.shape)

    df = df[df['labels']!=0]
    df_cleaned = df_cleaned[df_cleaned['labels']!=0]
    df_cleaned_char = df_cleaned_char[df_cleaned_char['labels']!=0]

    print(df.shape)
    print(df_cleaned.shape)
    print(df_cleaned_char.shape)

    df.to_csv(os.path.join(args.data_dir, 'raw_df.csv'), index=False)
    df_cleaned.to_csv(os.path.join(args.data_dir, 'cleaned_df.csv'), index=False)
    df_cleaned_char.to_csv(os.path.join(args.data_dir, 'cleaned_char_df.csv'), index=False)

    # КОРПУС
    corpus = []
    for col in columns[:-1]:
        corpus.extend(df_cleaned[col].tolist())

    corpus = list(set(corpus))

    corpus_char = []
    for col in columns[:-1]:
        corpus_char.extend(df_cleaned_char[col].tolist())

    corpus_char = list(set(corpus_char))

    print('getting vocabs...')
    words = ' '.join(corpus).split()
    unigrams_counter = Counter(words)

    bigrams = ngrams(words, 2)
    bigrams_counter = Counter(bigrams)

    char3grams = ngrams(' '.join(corpus_char), 3)
    char_counter = Counter(char3grams)

    print(len(unigrams_counter.most_common()))
    print(len(bigrams_counter.most_common()))
    print(len(char_counter.most_common()))

    uni2tok = [o for o, c in unigrams_counter.most_common(args.num_uni) if c > args.min_freq]
    uni2tok.insert(0, 'u_k')
    uni2idx = {tok: i + 1 for i, tok in enumerate(uni2tok)}

    bi2tok = [o for o, c in bigrams_counter.most_common(args.num_bi) if c > args.min_freq]
    bi2tok.insert(0, 'u_k')
    bi2idx = {tok: i + 1 for i, tok in enumerate(bi2tok)}

    char2tok = [o for o, c in char_counter.most_common(args.num_chars) if c > args.min_freq]
    char2tok.insert(0, 'u_k')
    char2idx = {tok: i + 1 for i, tok in enumerate(char2tok)}

    print('uni vocab len = {}'.format(len(uni2idx)))
    print('bi vocab len = {}'.format(len(bi2idx)))
    print('char vocab len = {}'.format(len(char2idx)))

    pickle.dump(uni2idx, open(uni2idx_path, 'wb'))
    pickle.dump(bi2idx, open(bi2idx_path, 'wb'))
    pickle.dump(char2idx, open(char2idx_path, 'wb'))
