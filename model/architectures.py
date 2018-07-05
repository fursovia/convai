
from model.attention_layer import attention
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import GRUCell
from model.model_utils import compute_embeddings
from model.attention_module import multihead_attention, layer_prepostprocess, shape_list
import tensorflow as tf
import tensorflow_hub as hub
from model.loss import _pairwise_distances


def build_model(is_training, sentences, params):
    """Compute logits of the model (output distribution)
    Args:
        mode: (string) 'train', 'eval', etc.
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)

    Returns:
        output: (tf.Tensor) output of the model
    """
    # def get_elmo(module, key):
    #     return hub.text_embedding_column(key=key, module_spec=module)

    if params.architecture == 'simple':
        elmo = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
        # history_emb = tf.expand_dims(elmo(history),1)
        # history = [sentences['quest'], sentences['cont']]
        with tf.variable_scope("model_elmo"):
            r = elmo(sentences['resp'])
        with tf.variable_scope("model_elmo", reuse=True):
            q = elmo(sentences['quest'])
        # with tf.variable_scope("model_elmo", reuse=True):
        #     c = elmo(sentences['cont'])
        with tf.variable_scope("model_elmo", reuse=True):
            f1 = elmo(sentences['fact1'])
        with tf.variable_scope("model_elmo", reuse=True):
            f2 = elmo(sentences['fact2'])
        with tf.variable_scope("model_elmo", reuse=True):
            f3 = elmo(sentences['fact3'])
        with tf.variable_scope("model_elmo", reuse=True):
            f4 = elmo(sentences['fact4'])
        with tf.variable_scope("model_elmo", reuse=True):
            f5 = elmo(sentences['fact5'])

        facts = tf.reshape(tf.concat([f1, f2, f3, f4, f5], axis=1), [-1, 5, 512])

        with tf.name_scope("closest_fact"):
            dot_product = tf.matmul(tf.expand_dims(q, 1), facts, transpose_b=True)  # [None, 5]
            dot_product = tf.reshape(dot_product, [-1, 5])
            max_dot_product = tf.reduce_max(dot_product, axis=1, keepdims=True)  # how close?
            max_fact_id = tf.argmax(dot_product, axis=1)
            mask = tf.cast(tf.one_hot(max_fact_id, 5), tf.bool)
            closest_info = tf.boolean_mask(facts, mask, axis=0)

        QR_sim = tf.sigmoid(tf.squeeze(tf.matmul(tf.expand_dims(q, 1), tf.expand_dims(r, -1))))
        #qrsim = tf.matmul(tf.expand_dims(q, 1), tf.expand_dims(r, -1))

        concatenated = tf.concat([q,
                                  r,
                                  f1,
                                  closest_info,
                                  max_dot_product], axis=1)

        with tf.variable_scope('fc_0'):
            dense0 = tf.layers.dense(concatenated, 1024, activation=tf.nn.relu)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(dense0, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3, QR_sim, q, r

    if params.architecture == 'simple2':
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
        # history_emb = tf.expand_dims(elmo(history),1)
        # history = [sentences['quest'], sentences['cont']]
        with tf.variable_scope("model_elmo"):
            r = elmo(sentences['resp'])
        with tf.variable_scope("model_elmo", reuse=True):
            q = elmo(sentences['quest'])
        # with tf.variable_scope("model_elmo", reuse=True):
        #     c = elmo(sentences['cont'])
        with tf.variable_scope("model_elmo", reuse=True):
            f1 = elmo(sentences['fact1'])
        with tf.variable_scope("model_elmo", reuse=True):
            f2 = elmo(sentences['fact2'])
        with tf.variable_scope("model_elmo", reuse=True):
            f3 = elmo(sentences['fact3'])
        with tf.variable_scope("model_elmo", reuse=True):
            f4 = elmo(sentences['fact4'])
        with tf.variable_scope("model_elmo", reuse=True):
            f5 = elmo(sentences['fact5'])

        facts = tf.reshape(tf.concat([f1, f2, f3, f4, f5], axis=1), [-1, 5, 1024])

        with tf.name_scope("closest_fact"):
            dot_product = tf.matmul(tf.expand_dims(q, 1), facts, transpose_b=True)  # [None, 5]
            dot_product = tf.reshape(dot_product, [-1, 5])
            max_dot_product = tf.reduce_max(dot_product, axis=1, keepdims=True)  # how close?
            max_fact_id = tf.argmax(dot_product, axis=1)
            mask = tf.cast(tf.one_hot(max_fact_id, 5), tf.bool)
            closest_info = tf.boolean_mask(facts, mask, axis=0)

        QR_sim = tf.sigmoid(tf.squeeze(tf.matmul(tf.expand_dims(q, 1), tf.expand_dims(r, -1))))
        #qrsim = tf.matmul(tf.expand_dims(q, 1), tf.expand_dims(r, -1))

        concatenated = tf.concat([q,
                                  r,
                                  f1,
                                  closest_info,
                                  max_dot_product], axis=1)

        with tf.variable_scope('fc_0'):
            dense0 = tf.layers.dense(concatenated, 1024, activation=tf.nn.relu)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(dense0, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3, QR_sim, q, r


    if params.architecture == 'smart':
        def gru_encoder(X):
            _, words_final_state = tf.nn.dynamic_rnn(cell=GRUCell(256),
                                                     inputs=X,
                                                     dtype=tf.float32)
            return words_final_state

        embeds_dict = compute_embeddings(sentences, params)

        context_u = embeds_dict['unigrams']['context']
        question_u = embeds_dict['unigrams']['question']
        response_u = embeds_dict['unigrams']['response']
        personal_info_u = embeds_dict['unigrams']['personal_info']

        info1 = tf.reshape(personal_info_u[:, :20], [-1, 20, params.embedding_size])
        info2 = tf.reshape(personal_info_u[:, 20:40], [-1, 20, params.embedding_size])
        info3 = tf.reshape(personal_info_u[:, 40:60], [-1, 20, params.embedding_size])
        info4 = tf.reshape(personal_info_u[:, 60:80], [-1, 20, params.embedding_size])
        info5 = tf.reshape(personal_info_u[:, 80:100], [-1, 20, params.embedding_size])

        with tf.variable_scope("GRU_encoder"):
            reply_gru = gru_encoder(response_u)

        with tf.variable_scope("GRU_encoder", reuse=True):
            question_gru = gru_encoder(question_u)

        with tf.variable_scope("GRU_encoder2"): #, reuse=True):
            #context_gru = gru_encoder(context_u)
            context_gru = tf.reduce_sum(context_u, axis=1)

        with tf.variable_scope("GRU_encoder", reuse=True):
            info_encoder1 = gru_encoder(info1)
            info_encoder2 = gru_encoder(info2)
            info_encoder3 = gru_encoder(info3)
            info_encoder4 = gru_encoder(info4)
            info_encoder5 = gru_encoder(info5)

        concatenated_info = tf.concat([info_encoder1,
                                       info_encoder2,
                                       info_encoder3,
                                       info_encoder4,
                                       info_encoder5], axis=1)

        reshaped_info = tf.reshape(concatenated_info, [-1, 5, 256])

        with tf.name_scope("closest_fact"):
            dot_product = tf.matmul(tf.expand_dims(question_gru, 1), reshaped_info, transpose_b=True)  # [None, 5]
            dot_product = tf.reshape(dot_product, [-1, 5])
            max_dot_product = tf.reduce_max(dot_product, axis=1, keepdims=True)  # how close?
            max_fact_id = tf.argmax(dot_product, axis=1)
            mask = tf.cast(tf.one_hot(max_fact_id, 5), tf.bool)
            closest_info = tf.boolean_mask(reshaped_info, mask, axis=0)

        QR_sim = tf.sigmoid(tf.squeeze(tf.matmul(tf.expand_dims(question_gru, 1), tf.expand_dims(reply_gru, -1))))

        concatenated = tf.concat([context_gru,
                                  question_gru,
                                  reply_gru,
                                  closest_info,
                                  max_dot_product], axis=1)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(concatenated, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3, QR_sim, question_u, reply_gru


    if params.architecture == 'smart2':
        embeds_dict = compute_embeddings(sentences, params)

        context_u = embeds_dict['unigrams']['context']
        question_u = embeds_dict['unigrams']['question']
        response_u = embeds_dict['unigrams']['response']
        personal_info_u = embeds_dict['unigrams']['personal_info']

        info1 = tf.reshape(personal_info_u[:, :20], [-1, 20, params.embedding_size])
        info2 = tf.reshape(personal_info_u[:, 20:40], [-1, 20, params.embedding_size])
        info3 = tf.reshape(personal_info_u[:, 40:60], [-1, 20, params.embedding_size])
        info4 = tf.reshape(personal_info_u[:, 60:80], [-1, 20, params.embedding_size])
        info5 = tf.reshape(personal_info_u[:, 80:100], [-1, 20, params.embedding_size])

        with tf.variable_scope("GRU_encoder"):
            reply_gru = tf.reduce_sum(response_u, axis=1)

        with tf.variable_scope("GRU_encoder", reuse=True):
            question_gru = tf.reduce_sum(question_u, axis=1)

        with tf.variable_scope("GRU_encoder", reuse=True):
            context_gru = tf.reduce_sum(context_u, axis=1)

        with tf.variable_scope("GRU_encoder", reuse=True):
            info_encoder1 = tf.reduce_sum(info1, axis=1)
            info_encoder2 = tf.reduce_sum(info2, axis=1)
            info_encoder3 = tf.reduce_sum(info3, axis=1)
            info_encoder4 = tf.reduce_sum(info4, axis=1)
            info_encoder5 = tf.reduce_sum(info5, axis=1)

        concatenated_info = tf.concat([info_encoder1,
                                       info_encoder2,
                                       info_encoder3,
                                       info_encoder4,
                                       info_encoder5], axis=1)

        reshaped_info = tf.reshape(concatenated_info, [-1, 5, 300])

        with tf.name_scope("closest_fact"):
            dot_product = tf.matmul(tf.expand_dims(question_gru, 1), reshaped_info, transpose_b=True)  # [None, 5]
            dot_product = tf.reshape(dot_product, [-1, 5])
            max_dot_product = tf.reduce_max(dot_product, axis=1, keepdims=True)  # how close?
            max_fact_id = tf.argmax(dot_product, axis=1)
            mask = tf.cast(tf.one_hot(max_fact_id, 5), tf.bool)
            closest_info = tf.boolean_mask(reshaped_info, mask, axis=0)

        QR_sim = tf.sigmoid(tf.squeeze(tf.matmul(tf.expand_dims(question_gru, 1), tf.expand_dims(reply_gru, -1))))

        concatenated = tf.concat([context_gru,
                                  question_gru,
                                  reply_gru,
                                  closest_info,
                                  max_dot_product], axis=1)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(concatenated, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3, QR_sim, question_u, reply_gru

    if params.architecture == 'elmo':
        elmo = hub.Module("https://tfhub.dev/google/elmo/2")  #, trainable=False)
        c = elmo(sentences['context'], signature='default', as_dict=True)['default']
        q = elmo(sentences['question'], signature='default', as_dict=True)['default']
        r = elmo(sentences['reply'], signature='default', as_dict=True)['default']
        f1 = elmo(sentences['fact1'], signature='default', as_dict=True)['default']
        f2 = elmo(sentences['fact2'], signature='default', as_dict=True)['default']
        f3 = elmo(sentences['fact3'], signature='default', as_dict=True)['default']
        f4 = elmo(sentences['fact4'], signature='default', as_dict=True)['default']
        f5 = elmo(sentences['fact5'], signature='default', as_dict=True)['default']
        facts = tf.reshape(tf.concat([f1, f2, f3, f4, f5], axis=1), [-1, 5, 1024])

        with tf.name_scope("closest_fact"):
            dot_product = tf.matmul(tf.expand_dims(q, 1), facts, transpose_b=True)  # [None, 5]
            dot_product = tf.reshape(dot_product, [-1, 5])
            max_dot_product = tf.reduce_max(dot_product, axis=1, keepdims=True)  # how close?
            max_fact_id = tf.argmax(dot_product, axis=1)
            mask = tf.cast(tf.one_hot(max_fact_id, 5), tf.bool)
            closest_info = tf.boolean_mask(facts, mask, axis=0)

        concatenated2 = tf.concat([c, q, r, closest_info, max_dot_product], axis=1)

        with tf.variable_scope('fc_0'):
            dense0 = tf.layers.dense(concatenated2, 1024, activation=tf.nn.relu)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(dense0, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3

    if params.architecture == 'elmo_0.1':
        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        # history_emb = tf.expand_dims(elmo(history),1)
        # history = [sentences['quest'], sentences['cont']]

        with tf.variable_scope("model"):
            reply_emb = elmo(sentences['resp'])
        with tf.variable_scope("model", reuse=True):
            quest_emb = elmo(sentences['quest'])
        with tf.variable_scope("model", reuse=True):
            cont_emb = elmo(sentences['cont'])
        with tf.variable_scope("model", reuse=True):
            fact1_emb = elmo(sentences['fact1'])
        with tf.variable_scope("model", reuse=True):
            fact2_emb = elmo(sentences['fact2'])
        with tf.variable_scope("model", reuse=True):
            fact3_emb = elmo(sentences['fact3'])
        with tf.variable_scope("model", reuse=True):
            fact4_emb = elmo(sentences['fact4'])
        with tf.variable_scope("model", reuse=True):
            fact5_emb = elmo(sentences['fact5'])

        print('fact1_emb', fact1_emb.shape)
        # personal_info_emb = tf.concat([tf.expand_dims(fact1_emb, 1), tf.expand_dims(fact2_emb, 1),
        #                                tf.expand_dims(fact3_emb, 1),
        #                                tf.expand_dims(fact4_emb, 1), tf.expand_dims(fact5_emb, 1)], axis=1)
        history_emb = tf.expand_dims(quest_emb, 1)
        # print('personal_info_emb', personal_info_emb.shape)
        print('history_emb', history_emb.shape)
        print('reply_emb', reply_emb.shape)

        # attention history on PI
        # with tf.variable_scope("self_attention"):
        #     # d_model = history_emb.shape[-1]
        #     # print('dim', d_model, personal_info_emb[-1])
        #     d_model = 1024
        #     y = multihead_attention(history_emb,
        #                             personal_info_emb,
        #                             d_model,
        #                             d_model,
        #                             d_model,
        #                             4,
        #                             name="multihead_attention_history_on_pi")
        #     history_emb = layer_prepostprocess(history_emb, y, 'a', 0., 'noam', d_model, 1e-6, 'normalization_attn')



        #temp
        quest_emb = tf.reduce_sum(history_emb, axis=1)
        print('quest_emb', quest_emb.shape)

        with tf.variable_scope('fc_3'):
            reply_emb = tf.layers.dense(reply_emb, 512)
            quest_emb = tf.layers.dense(quest_emb, 512)

        return quest_emb, reply_emb



        # reply_emb = elmo(sentences['resp'])
        # quest_emb = elmo(sentences['quest'])

        # concat = tf.concat([sentences['quest'], sentences['resp']], axis=0)
        # concat = elmo(concat)
        # bs = shape_list(sentences['quest'])[0]
        # reply_emb = concat[:bs]
        # quest_emb = concat[bs:]



    if params.architecture == 'first':
        # embeds_dict = compute_embeddings(sentences, params)
        #
        # context_u = embeds_dict['unigrams']['context']
        # question_u = embeds_dict['unigrams']['question']
        # response_u = embeds_dict['unigrams']['response']
        # personal_info_u = embeds_dict['unigrams']['personal_info']

        # 'context', 'question', 'reply', 'fact1', 'fact2', 'fact3', 'fact4',
        # 'fact5'
        # print('sentences', sentences['fact1'].shape)
        # print('question', sentences['question'].shape, sentences['context'].shape)

        # personal_info = tf.concat([sentences['fact1'], sentences['fact2'], sentences['fact3'], sentences['fact4']],
        #                           axis=-1)
        # history = tf.concat([sentences['question'], sentences['context']], axis=-1)
        # print('personal_info', personal_info.shape)
        # print('history', history.shape)

        # 'context', 'question', 'reply', 'fact1', 'fact2', 'fact3', 'fact4',
        # 'fact5'

        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        # history_emb = tf.expand_dims(elmo(history),1)

        reply_emb = elmo(sentences['reply'])
        # fact1 = tf.expand_dims(elmo(sentences['fact1']),1)
        # fact2 = tf.expand_dims(elmo(sentences['fact2']),1)
        # fact3 = tf.expand_dims(elmo(sentences['fact3']),1)
        # fact4 = tf.expand_dims(elmo(sentences['fact4']),1)
        # fact5 = tf.expand_dims(elmo(sentences['fact5']),1)
        # print('fact1', fact1.shape)
        # personal_info_emb = tf.concat([fact1,fact2,fact3,fact4,fact5], axis=1)

        # elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        # personal_info_emb = elmo(
        #     inputs={
        #         "tokens": personal_info,
        #         "sequence_len": personal_info.shape[-1]
        #     },
        #     signature="tokens",
        #     as_dict=True)["elmo"]
        # history_emb = elmo(
        #     inputs={
        #         "tokens": history,
        #         "sequence_len": history.shape[-1]
        #     },
        #     signature="tokens",
        #     as_dict=True)["elmo"]
        # reply_emb = elmo(
        #     inputs={
        #         "tokens": sentences['reply'],
        #         "sequence_len": sentences['reply'].shape[-1]
        #     },
        #     signature="tokens",
        #     as_dict=True)["elmo"]

        # personal_info_emb = elmo(personal_info)['elmo']
        # history_emb = elmo(history)['elmo']
        # reply_emb = elmo(sentences['reply'])['elmo']

        print('personal_info_emb', personal_info_emb.shape)
        print('history_emb', history_emb.shape)
        print('reply_emb', reply_emb.shape)

        # attention history on PI
        # with tf.variable_scope("self_attention"):
        #     # d_model = history_emb.shape[-1]
        #     # print('dim', d_model, personal_info_emb[-1])
        #     d_model = 1024
        #     y = multihead_attention(history_emb,
        #                             personal_info_emb,
        #                             d_model,
        #                             d_model,
        #                             d_model,
        #                             4,
        #                             name="multihead_attention_history_on_pi")
        #     history_new = layer_prepostprocess(history_emb, y, 'a', 0., 'noam', d_model, 1e-6, 'normalization_attn')



        #temp
        history = tf.reduce_sum(history_emb, axis=1)
        # response_u = tf.reduce_sum(reply_emb, axis=1)
        response_u = reply_emb
        personal_info_u = tf.reduce_sum(personal_info_emb, axis=1)
        print('personal_info_u', personal_info_u.shape)
        print('history', history.shape)
        print('response_u', response_u.shape)

        history_new = elmo(history_new)
        response = elmo(reply_emb)
        print('history_new', history_new.shape)
        print('response', response.shape)

        concatenated = tf.concat([response_u, history, personal_info_u], axis=1)
        print('concatenated', concatenated.shape)

        concatenated = tf.layers.flatten(concatenated)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(concatenated, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3

    if params.architecture == 'memory_nn':
        embeds_dict = compute_embeddings(sentences, params)

        context_u = embeds_dict['unigrams']['context']
        question_u = embeds_dict['unigrams']['question']
        response_u = embeds_dict['unigrams']['response']
        personal_info_u = embeds_dict['unigrams']['personal_info']

        history = tf.concat([context_u, question_u], axis=1)

        # attention history on PI
        with tf.variable_scope("self_attention"):
            d_model = 300 #history.shape[-1]
            y = multihead_attention(history,
                                    personal_info_u,
                                    d_model,
                                    300, #personal_info_u[-1],
                                    d_model,
                                    3,
                                    name="multihead_attention_history_on_pi")
            history = layer_prepostprocess(history, y, 'ad', 0., 'noam', d_model, 1e-6, 'normalization_attn')


        #temp
        history = tf.reduce_sum(history, axis=1)
        response = tf.reduce_sum(response_u, axis=1)
        # personal_info_u = tf.reduce_sum(personal_info_u, axis=1)
        # print('last', personal_info_u.shape)

        #compute simularity
        response_normalized = tf.nn.l2_normalize(response, axis=1)
        history_normalized = tf.nn.l2_normalize(history, axis=1)
        distance = tf.maximum(1 - tf.matmul(history_normalized, response_normalized, adjoint_b=True), 0.0)

        # response, history,
        concatenated = tf.concat([response, history, history*response], axis=1)
        print('concatenated', concatenated.shape)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(concatenated, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3, distance

    if params.architecture == 'memory_nn_batch-0.2':
        embeds_dict = compute_embeddings(sentences, params)

        context = embeds_dict['unigrams']['context']
        question = embeds_dict['unigrams']['question']
        response = embeds_dict['unigrams']['response']
        personal_info = embeds_dict['unigrams']['personal_info']

        # attention history on PI
        with tf.variable_scope("PI_attention"):
            d_model = 300 #history.shape[-1]
            y = multihead_attention(question,
                                    personal_info,
                                    d_model,
                                    300, #personal_info_u[-1],
                                    d_model,
                                    3,
                                    name="multihead_attention_history_on_pi")
            question_new = layer_prepostprocess(question, y, 'ad', 0., 'noam', d_model, 1e-6, 'normalization_attn')

        # attention history on PI
        with tf.variable_scope("context_attention"):
            d_model = 300 #history.shape[-1]
            y = multihead_attention(question,
                                    context,
                                    d_model,
                                    300, #personal_info_u[-1],
                                    d_model,
                                    3,
                                    name="multihead_attention_history_on_pi")
            question_new = layer_prepostprocess(question_new, y, 'ad', 0., 'noam', d_model, 1e-6, 'normalization_attn')

        #temp
        question = tf.reduce_sum(question_new, axis=1)
        response = tf.reduce_sum(response, axis=1)
        pairwise_dist = _pairwise_distances(0.0, question, response, params, False)

        return (question, response), pairwise_dist

    if params.architecture == 'memory_nn_batch':
        embeds_dict = compute_embeddings(sentences, params)

        context_u = embeds_dict['unigrams']['context']
        question_u = embeds_dict['unigrams']['question']
        response_u = embeds_dict['unigrams']['response']
        personal_info_u = embeds_dict['unigrams']['personal_info']

        history = tf.concat([context_u, question_u], axis=1)

        # attention history on PI
        with tf.variable_scope("self_attention"):
            d_model = 300  # history.shape[-1]
            y = multihead_attention(history,
                                    personal_info_u,
                                    d_model,
                                    300,  # personal_info_u[-1],
                                    d_model,
                                    3,
                                    name="multihead_attention_history_on_pi")
            history = layer_prepostprocess(history, y, 'ad', 0., 'noam', d_model, 1e-6, 'normalization_attn')

        # temp
        history = tf.reduce_sum(history, axis=1)
        response = tf.reduce_sum(response_u, axis=1)
        pairwise_dist = _pairwise_distances(0.0, history, response, params, False)

        return (history, response), pairwise_dist

    if params.architecture == 'second':
        embeds_dict = compute_embeddings(sentences, params)

        context_u = embeds_dict['unigrams']['context']
        question_u = embeds_dict['unigrams']['question']
        response_u = embeds_dict['unigrams']['response']
        personal_info_u = embeds_dict['unigrams']['personal_info']

        info1_u = tf.reshape(personal_info_u[:, :20], [-1, 20, params.embedding_size])
        info2_u = tf.reshape(personal_info_u[:, 20:40], [-1, 20, params.embedding_size])
        info3_u = tf.reshape(personal_info_u[:, 40:60], [-1, 20, params.embedding_size])
        info4_u = tf.reshape(personal_info_u[:, 60:80], [-1, 20, params.embedding_size])
        info5_u = tf.reshape(personal_info_u[:, 80:100], [-1, 20, params.embedding_size])

        # personal_info_u = tf.reshape(personal_info_u, [-1, 5, 20])

        def gru_encoder(X):
            _, words_final_state = tf.nn.dynamic_rnn(cell=GRUCell(256),
                                                     inputs=X,
                                                     dtype=tf.float32)
            return words_final_state

        with tf.variable_scope("GRU_encoder"):
            response_gru = gru_encoder(response_u)

        with tf.variable_scope("GRU_encoder", reuse=True):
            question_gru = gru_encoder(question_u)

        with tf.variable_scope("GRU_encoder", reuse=True):
            info_encoder1 = gru_encoder(info1_u)
            info_encoder2 = gru_encoder(info2_u)
            info_encoder3 = gru_encoder(info3_u)
            info_encoder4 = gru_encoder(info4_u)
            info_encoder5 = gru_encoder(info5_u)

            concatenated_info = tf.concat([info_encoder1,
                                           info_encoder2,
                                           info_encoder3,
                                           info_encoder4,
                                           info_encoder5], axis=1)

            reshaped_info = tf.reshape(concatenated_info, [-1, 5, 256])


        with tf.name_scope("closest_fact"):
            response_gru = tf.cast(response_gru, tf.float64)
            dot_product = tf.matmul(tf.expand_dims(response_gru, 1), reshaped_info, transpose_b=True)  # [None, 5]
            dot_product = tf.reshape(dot_product, [-1, 5])
            max_dot_product = tf.reduce_max(dot_product, axis=1, keepdims=True)  # how close?
            max_fact_id = tf.argmax(dot_product, axis=1)
            mask = tf.cast(tf.one_hot(max_fact_id, 5), tf.bool)
            closest_info = tf.boolean_mask(reshaped_info, mask, axis=0)

        with tf.name_scope("context_attention"):
            enc_out_chars, _ = tf.nn.bidirectional_dynamic_rnn(BasicLSTMCell(256),
                                                               BasicLSTMCell(256),
                                                               context_u,
                                                               dtype=tf.float32)

            context_output = attention(enc_out_chars, 50)

        concatenated = tf.concat([tf.cast(context_output, tf.float32),
                                  tf.cast(question_gru, tf.float32),
                                  tf.cast(response_gru, tf.float32),
                                  tf.cast(closest_info, tf.float32),
                                  tf.cast(max_dot_product, tf.float32)], axis=1)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(concatenated, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3