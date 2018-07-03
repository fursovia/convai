
from model.attention_layer import attention
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import GRUCell
from model.model_utils import compute_embeddings
from model.attention_module import multihead_attention, layer_prepostprocess
import tensorflow as tf
import tensorflow_hub as hub


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

    if params.architecture == 'first':
        # embeds_dict = compute_embeddings(sentences, params)
        #
        # context_u = embeds_dict['unigrams']['context']
        # question_u = embeds_dict['unigrams']['question']
        # response_u = embeds_dict['unigrams']['response']
        # personal_info_u = embeds_dict['unigrams']['personal_info']

        # 'context', 'question', 'reply', 'fact1', 'fact2', 'fact3', 'fact4',
        # 'fact5'
        print('sentences', sentences['fact1'].shape)
        print('question', sentences['question'].shape, sentences['context'].shape)

        # personal_info = tf.concat([sentences['fact1'], sentences['fact2'], sentences['fact3'], sentences['fact4']],
        #                           axis=-1)
        history = tf.concat([sentences['question'], sentences['context']], axis=-1)
        # print('personal_info', personal_info.shape)
        # print('history', history.shape)

        # 'context', 'question', 'reply', 'fact1', 'fact2', 'fact3', 'fact4',
        # 'fact5'

        elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
        history_emb = tf.expand_dims(elmo(history),1)
        reply_emb = elmo(sentences['reply'])
        fact1 = tf.expand_dims(elmo(sentences['fact1']),1)
        fact2 = tf.expand_dims(elmo(sentences['fact2']),1)
        fact3 = tf.expand_dims(elmo(sentences['fact3']),1)
        fact4 = tf.expand_dims(elmo(sentences['fact4']),1)
        fact5 = tf.expand_dims(elmo(sentences['fact5']),1)
        print('fact1', fact1.shape)
        personal_info_emb = tf.concat([fact1,fact2,fact3,fact4,fact5], axis=1)

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
        response_u = tf.reduce_sum(response_u, axis=1)
        personal_info_u = tf.reduce_sum(personal_info_u, axis=1)
        print('last', personal_info_u.shape)

        # response_u, history,
        concatenated = tf.concat([history*response_u], axis=1)
        print('concatenated', concatenated.shape)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(concatenated, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3

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