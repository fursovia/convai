"""model architecture"""

import tensorflow as tf
from model.architectures import build_model

def model_fn(features, labels, mode, params):

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    with tf.variable_scope('model'):
        logits, qr_sim, q_emb, r_emb = build_model(is_training, features, params)

    preds = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'y_prob': preds[:, 1],
                       'y_pred': tf.argmax(preds, axis=1),
                       'q_emb': q_emb,
                       'r_emb': r_emb}

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          export_outputs={
                                          'predict': tf.estimator.export.PredictOutput(predictions)
                                          })

    one_hot_labels = tf.one_hot(labels, 2)

    loss = 0.7 * tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)) + \
           0.3 * tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=qr_sim))

    if mode == tf.estimator.ModeKeys.EVAL:
        acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(preds, axis=1), name='acc')
        with tf.variable_scope("metrics"):
            eval_metric_ops = {"accuracy": (acc, acc_op)}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # tf.summary.scalar('accuracy', acc_op)

    global_step = tf.train.get_global_step()  # number of batches seen so far
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=params.learning_rate,
        optimizer=tf.train.AdamOptimizer()
    )

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
