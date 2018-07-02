"""model architecture"""

import tensorflow as tf
from model.attention_layer import attention
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import GRUCell
from model.model_utils import compute_embeddings


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
    embeds_dict = compute_embeddings(sentences, params)

    context_u = embeds_dict['unigrams']['context']
    question_u = ['unigrams']['question']
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

    concatenated = tf.concat([context_output, question_gru, response_gru, closest_info, max_dot_product], axis=1)

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
