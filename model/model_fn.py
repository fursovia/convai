"""model architecture"""

import tensorflow as tf


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
    context = sentences[:, :80]
    question = sentences[:, 80:120]
    reply = sentences[:, 120:160]
    personal_info = sentences[:, 160:360]

    weights_initializer = tf.truncated_normal_initializer(stddev=0.001)

    with tf.name_scope("embedding"):
        embedding_matrix = tf.get_variable("embedding_matrix", shape=[(params.vocab_size + 1), params.embedding_size],
                                           initializer=weights_initializer,
                                           trainable=True,
                                           dtype=tf.float32)

        context = tf.nn.embedding_lookup(embedding_matrix, context)
        question = tf.nn.embedding_lookup(embedding_matrix, question)
        reply = tf.nn.embedding_lookup(embedding_matrix, reply)
        personal_info = tf.nn.embedding_lookup(embedding_matrix, personal_info)
        personal_info = tf.reshape(personal_info, [-1, 5, 40, params.embedding_size])

        context = tf.reduce_mean(context, axis=1)  # [None, 300]
        question = tf.reduce_mean(question, axis=1)  # [None, 300]
        reply = tf.reduce_mean(reply, axis=1)  # [None, 300]
        personal_info = tf.reduce_mean(personal_info, axis=2)  # [None, 5, 300]

    with tf.name_scope("closest_fact"):
        dot_product = tf.matmul(tf.expand_dims(reply, 1), personal_info, transpose_b=True)  # [None, 5]
        dot_product = tf.reshape(dot_product, [-1, 5])
        max_fact_id = tf.argmax(dot_product, axis=1)
        mask = tf.cast(tf.one_hot(max_fact_id, 5), tf.bool)
        closest_info = tf.boolean_mask(personal_info, mask, axis=0)

    concatenated = tf.concat([context, question, reply, closest_info], axis=1)

    with tf.variable_scope('fc_0'):
        dense0 = tf.layers.dense(concatenated, 1024, activation=tf.nn.relu)

    with tf.variable_scope('fc_1'):
        dense1 = tf.layers.dense(dense0, 512, activation=tf.nn.relu)

    with tf.variable_scope('fc_2'):
        dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)

    with tf.variable_scope('fc_3'):
        dense3 = tf.layers.dense(dense2, 1, activation=tf.nn.sigmoid)

    return dense3


def model_fn(features, labels, mode, params):
    """Model function defining the graph operations.

    Args:
        mode: (string) 'train', 'eval', etc.
        features: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        labels: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('model'):
        preds = build_model(is_training, features, params)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'predictions': preds}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=preds)
    acc = tf.metrics.accuracy(labels=labels, predictions=preds, name='acc')

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.variable_scope("metrics"):
            eval_metric_ops = {"accuracy": acc}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    tf.summary.scalar('loss', loss)
    # tf.summary.scalar('accuracy', tf.metrics.mean(acc[1]))

    optimizer = tf.train.AdamOptimizer(params.learning_rate)

    global_step = tf.train.get_global_step()

    # train_op = optimizer.minimize(loss, global_step=global_step)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=params.learning_rate,
        optimizer=optimizer
    )

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
