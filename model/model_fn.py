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
    pass


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

    # loss = ...
    # optimizer = ...

    pass
