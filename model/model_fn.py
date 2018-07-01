"""model architecture"""

import tensorflow as tf
from model.attention_layer import attention
from model.utils import get_embeddings
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import GRUCell
from .attention_module import multihead_attention


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
    context = sentences['cont']
    question = sentences['quest']
    reply = sentences['resp']
    personal_info = sentences['facts']
    # personal_info = tf.reshape(personal_info, [-1, 5, 140])

    cont_u = context[:, :20]
    quest_u = context[:, :20]
    resp_u = context[:, :20]
    facts_u = context[:, :20]

    cont_b = context[:, 20:40]
    quest_b = context[:, 20:40]
    resp_b = context[:, 20:40]
    facts_b = context[:, 20:40]

    cont_c = context[:, 40:140]
    quest_c = context[:, 40:140]
    resp_c = context[:, 40:140]
    facts_c = context[:, 40:140]

    weights_initializer = tf.truncated_normal_initializer(stddev=0.001)

    if params.architecture == 1:
        def gru_encoder(X):
            _, words_final_state = tf.nn.dynamic_rnn(cell=GRUCell(256),
                                                     inputs=X,
                                                     dtype=tf.float32)
            return words_final_state

        with tf.name_scope("embedding_words"):
            embedding_matrix_u = tf.get_variable("embedding_matrix_u",
                                                 shape=[(params.uni_size + 1), params.embedding_size],
                                                 initializer=weights_initializer,
                                                 trainable=True,
                                                 dtype=tf.float64)

            cont_u = tf.nn.embedding_lookup(embedding_matrix_u, cont_u)
            quest_u = tf.nn.embedding_lookup(embedding_matrix_u, quest_u)
            resp_u = tf.nn.embedding_lookup(embedding_matrix_u, resp_u)
            facts_u = tf.nn.embedding_lookup(embedding_matrix_u, facts_u)

        with tf.name_scope("embedding_bigrams"):
            embedding_matrix_b = tf.get_variable("embedding_matrix_b",
                                                 shape=[(params.bi_size + 1), params.embedding_size],
                                                 initializer=weights_initializer,
                                                 trainable=True,
                                                 dtype=tf.float64)

            cont_b = tf.nn.embedding_lookup(embedding_matrix_b, cont_b)
            quest_b = tf.nn.embedding_lookup(embedding_matrix_b, quest_b)
            resp_b = tf.nn.embedding_lookup(embedding_matrix_b, resp_b)
            facts_b = tf.nn.embedding_lookup(embedding_matrix_b, facts_b)

        with tf.name_scope("embedding_chars"):
            embedding_matrix_c = tf.get_variable("embedding_matrix_c",
                                                 shape=[(params.char_size + 1), params.embedding_size],
                                                 initializer=weights_initializer,
                                                 trainable=True,
                                                 dtype=tf.float64)

            cont_c = tf.nn.embedding_lookup(embedding_matrix_c, cont_c)
            quest_c = tf.nn.embedding_lookup(embedding_matrix_c, quest_c)
            resp_c = tf.nn.embedding_lookup(embedding_matrix_c, resp_c)
            facts_c = tf.nn.embedding_lookup(embedding_matrix_c, facts_c)

        with tf.variable_scope("GRU_encoder"):
            reply_gru = gru_encoder(reply)

        with tf.variable_scope("GRU_encoder", reuse=True):
            question_gru = gru_encoder(question)

        with tf.variable_scope("GRU_encoder", reuse=True):
            info_encoder1 = gru_encoder(info1)
            info_encoder2 = gru_encoder(info2)
            info_encoder3 = gru_encoder(info3)
            info_encoder4 = gru_encoder(info4)
            info_encoder5 = gru_encoder(info5)

            concatenated_info = tf.concat([info_encoder1, info_encoder2, info_encoder3, info_encoder4, info_encoder5], axis=1)
            reshaped_info = tf.reshape(concatenated_info, [-1, 5, 256])

        with tf.name_scope("closest_fact"):
            dot_product = tf.matmul(tf.expand_dims(reply_gru, 1), reshaped_info, transpose_b=True)  # [None, 5]
            dot_product = tf.reshape(dot_product, [-1, 5])
            max_dot_product = tf.reduce_max(dot_product, axis=1, keepdims=True)  # how close?
            max_fact_id = tf.argmax(dot_product, axis=1)
            mask = tf.cast(tf.one_hot(max_fact_id, 5), tf.bool)
            closest_info = tf.boolean_mask(reshaped_info, mask, axis=0)

        with tf.name_scope("context_attention"):
            enc_out_chars, _ = tf.nn.bidirectional_dynamic_rnn(BasicLSTMCell(256),
                                                               BasicLSTMCell(256),
                                                               context,
                                                               dtype=tf.float32)

            context_output = attention(enc_out_chars, 50)

        concatenated = tf.concat([context_output, question_gru, reply_gru, closest_info, max_dot_product], axis=1)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(concatenated, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3

    if params.architecture == 2:
        def gru_encoder(X):
            _, words_final_state = tf.nn.dynamic_rnn(cell=GRUCell(256),
                                                     inputs=X,
                                                     dtype=tf.float32)
            return words_final_state

        with tf.name_scope("embedding"):
            embedding_matrix = tf.get_variable("embedding_matrix", shape=[(params.vocab_size + 1), params.embedding_size],
                                               initializer=weights_initializer,
                                               trainable=True,
                                               dtype=tf.float32)

            context = tf.nn.embedding_lookup(embedding_matrix, context)
            question = tf.nn.embedding_lookup(embedding_matrix, question)
            reply = tf.nn.embedding_lookup(embedding_matrix, reply)
            personal_info = tf.nn.embedding_lookup(embedding_matrix, personal_info)

        with tf.name_scope("multihead_attention"):
            context = multihead_attention(context, context, context.shape[-1], context.shape[-1], context.shape[-1], 3)
#             question = multihead_attention(context, context, context.shape[-1], context.shape[-1], context.shape[-1], 3)
#             reply = multihead_attention(context, context, context.shape[-1], context.shape[-1], context.shape[-1], 3)
#             personal_info = multihead_attention(context, context, context.shape[-1], context.shape[-1], context.shape[-1], 3)

        with tf.variable_scope("GRU_encoder"):
            reply_gru = gru_encoder(reply)

        with tf.variable_scope("GRU_encoder", reuse=True):
            question_gru = gru_encoder(question)

        with tf.variable_scope("GRU_encoder", reuse=True):
            info_encoder1 = gru_encoder(info1)
            info_encoder2 = gru_encoder(info2)
            info_encoder3 = gru_encoder(info3)
            info_encoder4 = gru_encoder(info4)
            info_encoder5 = gru_encoder(info5)

            concatenated_info = tf.concat([info_encoder1, info_encoder2, info_encoder3, info_encoder4, info_encoder5], axis=1)
            reshaped_info = tf.reshape(concatenated_info, [-1, 5, 256])


        with tf.name_scope("context_attention"):
            enc_out_chars, _ = tf.nn.bidirectional_dynamic_rnn(BasicLSTMCell(256),
                                                               BasicLSTMCell(256),
                                                               context,
                                                               dtype=tf.float32)

            context_output = attention(enc_out_chars, 50)

        concatenated = tf.concat([context_output, question_gru, reply_gru, closest_info, max_dot_product], axis=1)

        with tf.variable_scope('fc_1'):
            dense1 = tf.layers.dense(concatenated, 512, activation=tf.nn.relu)

        with tf.variable_scope('fc_2'):
            dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

        with tf.variable_scope('fc_3'):
            dense3 = tf.layers.dense(dense2, 2)

        return dense3


def model_fn(features, labels, mode, params):

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('model'):
        logits = build_model(is_training, features, params)

    preds = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'y_prob': preds[:, 1],
                       'y_pred': tf.argmax(preds, axis=1)}

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          export_outputs={
                                              'predict': tf.estimator.export.PredictOutput(predictions)
                                          })

    one_hot_labels = tf.one_hot(labels, 2)

    loss = tf.reduce_mean(
        tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
    )

    acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(preds, axis=1), name='acc')

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.variable_scope("metrics"):
            eval_metric_ops = {"accuracy": (acc, acc_op)}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    tf.summary.scalar('accuracy', acc_op)

    global_step = tf.train.get_global_step()  # number of batches seen so far
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=params.learning_rate,
        optimizer=tf.train.AdamOptimizer()
    )

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
