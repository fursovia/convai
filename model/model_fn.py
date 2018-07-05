"""model architecture"""

import tensorflow as tf
from model.architectures import build_model
from model.loss import get_loss

def model_fn(features, labels, mode, params):

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    if params.loss_type == 'usual':

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


        if mode == tf.estimator.ModeKeys.EVAL:
            acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(preds, axis=1), name='acc')
            with tf.variable_scope("metrics"):
                eval_metric_ops = {"accuracy": (acc, acc_op)}

            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

        #tf.summary.scalar('accuracy', acc_op)

    else:
        with tf.variable_scope('model'):
            logits, dists = build_model(is_training, features, params)
            # qr_sim, q_emb, r_emb
            print('logits', logits[0].shape, logits[0].shape)


        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'hist_emb': logits[0],
                           'resp_emb': logits[1],
                           'dists': dists}

            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              export_outputs={
                                                  'predict': tf.estimator.export.PredictOutput(predictions)
                                              })

        loss, fraction, p_at_k, mrr = get_loss(labels, logits, params)
        if mode == tf.estimator.ModeKeys.EVAL:
            with tf.variable_scope("metrics"):
                eval_metric_ops = {"precision_at_K": tf.metrics.mean(p_at_k),
                                   "MRR": tf.metrics.mean(mrr),
                                   "fraction_positive_triplets": tf.metrics.mean(fraction)}

            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('fraction_positive_triplets', fraction)
        tf.summary.scalar('precision_at_K', p_at_k)
        tf.summary.scalar('MRR', mrr)

        # tf.summary.scalar('accuracy', acc_op)

    global_step = tf.train.get_global_step()  # number of batches seen so far
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=params.learning_rate,
        optimizer=tf.train.AdamOptimizer()
    )

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
