import tensorflow as tf
import numpy as np
import pickle
import os

from model.utils import get_embeddings


def get_coefs(word, *arr):
    return word, np.array(arr, dtype="float32")


def get_embeddings2(dict_dir='/data/dssm_data/dictionaries',
                    emb_dir='/data/i.anokhin/world_embeddings'):
    """Создает матрицу с предобученными эмбедингами
    Args:
        dict_dir: path to the dictionary
        emb_dir: path to the pretrained fasttext .vec file
    Returns:
        embedding_matrix: матрица с предобученными эмбедингами с помощью fasttext
    """
    dict_dir = os.path.join()
    assert os.path.isfile(dict_dir), "No word2idx file found at {}".format(dict_dir)
    assert os.path.isfile(emb_dir), "No embedding file found at {}".format(emb_dir)

    word2idx = pickle.load(open(dict_dir, 'rb'))
    embeddings_index = dict(
        get_coefs(*o.strip().split()) for o in open(emb_dir, encoding='utf-8') if o.strip().split()[0] in word2idx)

    vocab_size = len(word2idx)
    embedding_size = list(embeddings_index.values())[1].shape[0]

    embedding_matrix = np.zeros((vocab_size, embedding_size))

    cnt = 0
    for word, i in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, embedding_size)
            cnt += 1

    print('number of unknown words: ', cnt)

    return embedding_matrix

def compute_embeddings(sentences, params):
    what_to_get = params.embeds  # ubc (unigrams, bigrams, chars)
    embeds_to_return = {}

    weights_initializer = get_embeddings2(dict_dir=params.dict_paths, emb_dir=params.emb_dir)

    # weights_initializer = tf.truncated_normal_initializer(stddev=0.001)

    context = sentences['cont']
    question = sentences['quest']
    reply = sentences['resp']
    personal_info = sentences['facts']
    # personal_info = tf.reshape(personal_info, [-1, 5, 140])

    cont_u = context[:, :20]
    quest_u = question[:, :20]
    resp_u = reply[:, :20]
    facts_u = personal_info[:, :20]

    cont_b = context[:, 20:40]
    quest_b = question[:, 20:40]
    resp_b = reply[:, 20:40]
    facts_b = personal_info[:, 20:40]

    cont_c = context[:, 40:140]
    quest_c = question[:, 40:140]
    resp_c = reply[:, 40:140]
    facts_c = personal_info[:, 40:140]

    if what_to_get == 'u':
        with tf.name_scope("embedding_words"):
            embedding_matrix_u = tf.get_variable("embedding_matrix_u",
                                                 shape=[(params.uni_size + 1), params.embedding_size],
                                                 initializer=weights_initializer,
                                                 trainable=True,
                                                 dtype=tf.float32)

            cont_u = tf.nn.embedding_lookup(embedding_matrix_u, cont_u)
            quest_u = tf.nn.embedding_lookup(embedding_matrix_u, quest_u)
            resp_u = tf.nn.embedding_lookup(embedding_matrix_u, resp_u)
            facts_u = tf.nn.embedding_lookup(embedding_matrix_u, facts_u)

        embeds_to_return['unigrams'] = {}
        embeds_to_return['unigrams']['context'] = cont_u
        embeds_to_return['unigrams']['question'] = quest_u
        embeds_to_return['unigrams']['response'] = resp_u
        embeds_to_return['unigrams']['personal_info'] = facts_u

        return embeds_to_return

    if what_to_get == 'ub':
        with tf.name_scope("embedding_words"):
            embedding_matrix_u = tf.get_variable("embedding_matrix_u",
                                                 shape=[(params.uni_size + 1), params.embedding_size],
                                                 initializer=weights_initializer,
                                                 trainable=True,
                                                 dtype=tf.float32)

            cont_u = tf.nn.embedding_lookup(embedding_matrix_u, cont_u)
            quest_u = tf.nn.embedding_lookup(embedding_matrix_u, quest_u)
            resp_u = tf.nn.embedding_lookup(embedding_matrix_u, resp_u)
            facts_u = tf.nn.embedding_lookup(embedding_matrix_u, facts_u)

        with tf.name_scope("embedding_bigrams"):
            embedding_matrix_b = tf.get_variable("embedding_matrix_b",
                                                 shape=[(params.bi_size + 1), params.embedding_size],
                                                 initializer=weights_initializer,
                                                 trainable=True,
                                                 dtype=tf.float32)

            cont_b = tf.nn.embedding_lookup(embedding_matrix_b, cont_b)
            quest_b = tf.nn.embedding_lookup(embedding_matrix_b, quest_b)
            resp_b = tf.nn.embedding_lookup(embedding_matrix_b, resp_b)
            facts_b = tf.nn.embedding_lookup(embedding_matrix_b, facts_b)

        embeds_to_return['unigrams'] = {}
        embeds_to_return['bigrams'] = {}
        embeds_to_return['unigrams']['context'] = cont_u
        embeds_to_return['unigrams']['question'] = quest_u
        embeds_to_return['unigrams']['response'] = resp_u
        embeds_to_return['unigrams']['personal_info'] = facts_u

        embeds_to_return['bigrams']['context'] = cont_b
        embeds_to_return['bigrams']['question'] = quest_b
        embeds_to_return['bigrams']['response'] = resp_b
        embeds_to_return['bigrams']['personal_info'] = facts_b

        return embeds_to_return

    if what_to_get == 'ubc':
        with tf.name_scope("embedding_words"):
            embedding_matrix_u = tf.get_variable("embedding_matrix_u",
                                                 shape=[(params.uni_size + 1), params.embedding_size],
                                                 initializer=weights_initializer,
                                                 trainable=True,
                                                 dtype=tf.float32)

            cont_u = tf.nn.embedding_lookup(embedding_matrix_u, cont_u)
            quest_u = tf.nn.embedding_lookup(embedding_matrix_u, quest_u)
            resp_u = tf.nn.embedding_lookup(embedding_matrix_u, resp_u)
            facts_u = tf.nn.embedding_lookup(embedding_matrix_u, facts_u)

        with tf.name_scope("embedding_bigrams"):
            embedding_matrix_b = tf.get_variable("embedding_matrix_b",
                                                 shape=[(params.bi_size + 1), params.embedding_size],
                                                 initializer=weights_initializer,
                                                 trainable=True,
                                                 dtype=tf.float32)

            cont_b = tf.nn.embedding_lookup(embedding_matrix_b, cont_b)
            quest_b = tf.nn.embedding_lookup(embedding_matrix_b, quest_b)
            resp_b = tf.nn.embedding_lookup(embedding_matrix_b, resp_b)
            facts_b = tf.nn.embedding_lookup(embedding_matrix_b, facts_b)

        with tf.name_scope("embedding_chars"):
            embedding_matrix_c = tf.get_variable("embedding_matrix_c",
                                                 shape=[(params.char_size + 1), params.embedding_size],
                                                 initializer=weights_initializer,
                                                 trainable=True,
                                                 dtype=tf.float32)

            cont_c = tf.nn.embedding_lookup(embedding_matrix_c, cont_c)
            quest_c = tf.nn.embedding_lookup(embedding_matrix_c, quest_c)
            resp_c = tf.nn.embedding_lookup(embedding_matrix_c, resp_c)
            facts_c = tf.nn.embedding_lookup(embedding_matrix_c, facts_c)

        embeds_to_return['unigrams'] = {}
        embeds_to_return['bigrams'] = {}
        embeds_to_return['chars'] = {}
        embeds_to_return['unigrams']['context'] = cont_u
        embeds_to_return['unigrams']['question'] = quest_u
        embeds_to_return['unigrams']['response'] = resp_u
        embeds_to_return['unigrams']['personal_info'] = facts_u

        embeds_to_return['bigrams']['context'] = cont_b
        embeds_to_return['bigrams']['question'] = quest_b
        embeds_to_return['bigrams']['response'] = resp_b
        embeds_to_return['bigrams']['personal_info'] = facts_b

        embeds_to_return['chars']['context'] = cont_c
        embeds_to_return['chars']['question'] = quest_c
        embeds_to_return['chars']['response'] = resp_c
        embeds_to_return['chars']['personal_info'] = facts_c

        return embeds_to_return

    if what_to_get == 'c':
        with tf.name_scope("embedding_chars"):
            embedding_matrix_c = tf.get_variable("embedding_matrix_c",
                                                 shape=[(params.char_size + 1), params.embedding_size],
                                                 initializer=weights_initializer,
                                                 trainable=True,
                                                 dtype=tf.float32)

            cont_c = tf.nn.embedding_lookup(embedding_matrix_c, cont_c)
            quest_c = tf.nn.embedding_lookup(embedding_matrix_c, quest_c)
            resp_c = tf.nn.embedding_lookup(embedding_matrix_c, resp_c)
            facts_c = tf.nn.embedding_lookup(embedding_matrix_c, facts_c)

        embeds_to_return['chars'] = {}
        embeds_to_return['chars']['context'] = cont_c
        embeds_to_return['chars']['question'] = quest_c
        embeds_to_return['chars']['response'] = resp_c
        embeds_to_return['chars']['personal_info'] = facts_c

        return embeds_to_return
