import tensorflow as tf
import numpy as np

debug = False


def kmax_pooling(conv, kmax, sort=True):
    """
    Take kmax maximum value for each filter.
      Args:
        conv: A Tensor to be transformed. (bath, seq_len, filters)
        kmax: an integer (number of maximum values to return for each filter)
        sort: wheather to sort maximum values or not
      Returns:
        a Tensor (bath, kmax, filters)
    """
    conv_dim = list(range(len(shape_list(conv))))
    with tf.name_scope('kmax_pool'):
        pool = tf.transpose(conv, perm=[0] + conv_dim[2:] + [1])
        pool = tf.nn.top_k(pool, k=kmax, sorted=sort)[0]
        pool = tf.transpose(pool, perm=[0, conv_dim[-1]] + conv_dim[1:-1])

    return pool


def top_interactions(x, kmax):
    """
      Choose top kmax row (dim=1) based on the first column (bath, seq_len, filters[0])
      Args:
        x: A Tensor to be transformed. (bath, seq_len, filters)
        kmax: number interaction to choose
      Returns:
        a Tensor (bath, kmax, filters)
    """
    # consider the first row
    dim = list(range(len(x.shape)))
    x_first = tf.transpose(x[:, :, :1], perm=[0] + dim[2:] + [1])
    if debug: print('x_first', x_first)
    value, ind = tf.nn.top_k(x_first, k=kmax, sorted=False)
    if debug: print('ind', ind)
    out = chose_rows_given_indices(x, ind)

    return out


def chose_rows_given_indices(x, ind):
    """
      Choose top kmax row (dim=1) based on the first column (bath, seq_len, filters[0])
      Args:
        x: A Tensor to be transformed. (bath, seq_len, filters)
        ind: row index (dim=1) to choose from the x tensor ((bath, number_indices_to_choose, 2))
      Returns:
        a Tensor (bath, ind.shape[1], filters)
    """
    # forming batch insex
    ind_shape = shape_list(ind)
    # ind_batches = tf.constant([list(range(512))] * (ind_shape[1] * ind_shape[2]))
    tf_range = tf.expand_dims(tf.range(0, ind_shape[0], 1), 0)
    ind_batches = tf.concat([tf_range] * (ind_shape[1] * ind_shape[2]), axis=0)
    ind_batches = tf.transpose(ind_batches, [1, 0])
    ind_batches = tf.expand_dims(ind_batches, 1)
    if debug: print('ind_batches', ind_batches)
    ind2 = tf.expand_dims(tf.contrib.layers.flatten(ind), 1)
    if debug: print('ind2', ind2)
    ind3 = tf.transpose(tf.concat([ind_batches, ind2], axis=1), [0, 2, 1])
    if debug: print('ind3', ind3)
    out = tf.gather_nd(x, ind3)
    if debug: print('out', out)

    return out


def convolution_on_attention(query_antecedent, memory_antecedent, attention_kmax=2, num_filters=5,
                             activation=None, convolution_kmax=10, name=None):
    """
      Perform convolution on choosen by attention mechanism words(dim=1), after that choose top convolution interations.
      Args:
        query_antecedent: a Tensor with shape (batch, seq, filters)
        memory_antecedent: a Tensor with shape (batch, seq, filters)
        attention_kmax: int (number of indices to choose to perform convolution on)
        num_filters: int
        activation: tf activation for after convolution
        convolution_kmax: int (number of top convolution interations to choose)
        name: string
      Returns:
        a Tensor (bath, ind.shape[1], filters)
    """
    total_key_depth = query_antecedent.shape[-1]
    q = compute_attention_component(query_antecedent, total_key_depth, "q")
    k = compute_attention_component(query_antecedent, total_key_depth, "k")
    v = memory_antecedent
    with tf.variable_scope(
            name, default_name="conv_on_attn", values=[q, k, v]) as scope:
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits, name="attention_weights")
        weights_value, weights_idx = tf.nn.top_k(weights, k=attention_kmax, sorted=False)
        out = chose_rows_given_indices(v, weights_idx)
        conv = tf.layers.conv1d(
            inputs=out,
            filters=num_filters,
            kernel_size=attention_kmax,
            strides=attention_kmax,
            padding="valid",
            activation=activation
        )
        if debug: print('conv', conv)
        # out = top_interactions(conv, kmax=convolution_kmax)
        # out = conv
        out = kmax_pooling(conv, kmax=convolution_kmax)
        return out


def transformer_encoder_layer(x, num_heads, num_hidden_layers):
    """
Apply transformer encoder to the sequence.
  Args:
    x: A Tensor to be transformed. (batch, seq, filters)
    num_heads: an integer (number of heads in multiheadattention)
    num_hidden_layers: an integer
  Returns:
    a Tensor (batch, seq, filters)
    """
    d_model = x.shape[-1]
    for layer in range(num_hidden_layers):
        with tf.variable_scope("layer_%d" % layer):
            with tf.variable_scope("self_attention"):
                y = multihead_attention(x, x, d_model, d_model, d_model, num_heads, )
                x = layer_prepostprocess(x, y, 'and', 0., 'noam', d_model, 1e-6, 'normalization_attn')
            with tf.variable_scope("ffn"):
                y = dense_relu_dense(x, 4 * d_model, d_model, dropout=0.1)
                x = layer_prepostprocess(x, y, 'and', 0., 'noam', d_model, 1e-6, 'normalization_ff')
    return x


def dense_relu_dense(inputs,
                     filter_size,
                     output_size,
                     output_activation=None,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     name=None):
    """Hidden layer with RELU activation followed by linear projection."""
    layer_name = "%s_{}" % name if name else "{}"
    h = tf.layers.dense(inputs,
                        filter_size,
                        use_bias=True,
                        activation=tf.nn.relu,
                        name=layer_name.format("conv1"))

    if dropout != 0.0:
        h = dropout_with_broadcast_dims(
            h, 1.0 - dropout, broadcast_dims=dropout_broadcast_dims)
    o = tf.layers.dense(
        h,
        output_size,
        activation=output_activation,
        use_bias=True,
        name=layer_name.format("conv2"))

    return o


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
    """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.
      Instead of specifying noise_shape, this function takes broadcast_dims -
      a list of dimension numbers in which noise_shape should be 1.  The random
      keep/drop tensor has dimensionality 1 along these dimensions.
      Args:
        x: a floating point tensor.
        keep_prob: A scalar Tensor with the same type as x.
          The probability that each element is kept.
        broadcast_dims: an optional list of integers
          the dimensions along which to broadcast the keep/drop flags.
        **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".
      Returns:
        A Tensor with the same size and shape as x.
    """
    assert "noise_shape" not in kwargs
    if broadcast_dims:
        shape = tf.shape(x)
        ndims = len(x.get_shape())
        # Allow dimensions like "-1" as well.
        broadcast_dims = [dim + ndims if dim < 0 else dim for dim in broadcast_dims]
        kwargs["noise_shape"] = [
            1 if i in broadcast_dims else shape[i] for i in range(ndims)]

    return tf.nn.dropout(x, keep_prob, **kwargs)


def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         norm_type,
                         depth,
                         epsilon,
                         default_name,
                         name=None,
                         dropout_broadcast_dims=None):
    """Apply a sequence of functions to the input or output of a layer.
      The sequence is specified as a string which may contain the following
      characters:
        a: add previous_value
        n: apply normalization
        d: apply dropout
      For example, if sequence=="dna", then the output is
        previous_value + normalize(dropout(x))
      Args:
        previous_value: A Tensor, to be added as a residual connection ('a')
        x: A Tensor to be transformed.
        sequence: a string.
        dropout_rate: a float
        norm_type: a string (see apply_norm())
        depth: an integer (size of last dimension of x).
        epsilon: a float (parameter for normalization)
        default_name: a string
        name: a string
        dropout_broadcast_dims:  an optional list of integers less than 3
          specifying in which dimensions to broadcast the dropout decisions.
          saves memory.
      Returns:
        a Tensor

    """
    with tf.variable_scope(name, default_name=default_name):
        if sequence == "none":
            return x
        for c in sequence:
            if c == "a":
                x += previous_value
            elif c == "n":
                x = apply_norm(x, norm_type, depth, epsilon)
            else:
                assert c == "d", ("Unknown sequence step %s" % c)
                x = dropout_with_broadcast_dims(
                    x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
        return x


def apply_norm(x, norm_type, depth, epsilon):
    """Apply Normalization."""
    if norm_type == "batch":
        return tf.layers.batch_normalization(x, epsilon=epsilon)
    if norm_type == "noam":
        return noam_norm(x, epsilon)
    if norm_type == "none":
        return x

    raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                     "'noam', 'none'.")


def noam_norm(x, epsilon=1.0, name=None):
    """One version of layer normalization."""
    with tf.name_scope(name, default_name="noam_norm", values=[x]):
        shape = x.get_shape()
        ndims = len(shape)
        return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(
            tf.to_float(shape[-1])))


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        name="multihead_attention",
                        **kwargs):
    """
      Multihead scaled-dot-product attention with input/output transformations.
      Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
        bias: bias Tensor (see attention_bias())
        total_key_depth: an integer
        total_value_depth: an integer
        output_depth: an integer
        num_heads: an integer dividing total_key_depth and total_value_depth
      Returns:
        The result of the attention transformation. The output shape is
            [batch_size, length_q, hidden_dim]
      Raises:
        ValueError: if the key depth or value depth are not divisible by the
          number of attention heads.
    """
    if total_key_depth % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_key_depth, num_heads))
    if total_value_depth % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_value_depth, num_heads))
    with tf.variable_scope(name, default_name="multihead_attention",
                           values=[query_antecedent, memory_antecedent]):

        q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                              total_key_depth, total_value_depth, )

        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        q = split_heads(q, num_heads)

        key_depth_per_head = total_key_depth // num_heads

        x = dot_product_attention(
            q,
            k,
            v, )

        x = combine_heads(x)

        # Set last dim specifically.
        x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

        x = tf.layers.dense(
            x, output_depth, use_bias=False, name="output_transform")

        return x


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def combine_heads(x):
    """Inverse of split_heads.
      Args:
        x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
      Returns:
        a Tensor with shape [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
      Args:
        x: a Tensor with shape [..., a, b]
      Returns:
        a Tensor with shape [..., ab]
    """
    x_shape = shape_list(x)
    a, b = x_shape[-2:]
    return tf.reshape(x, x_shape[:-2] + [a * b])


def dot_product_attention(q,
                          k,
                          v,
                          name=None, ):
    """dot-product attention.
      Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        name: string
      Returns:
        A Tensor.
    """
    with tf.variable_scope(
            name, default_name="dot_product_attention", values=[q, k, v]) as scope:
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits, name="attention_weights")
        print('weights', weights.shape)
        new_q = tf.matmul(weights, v)
        return new_q


def split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).
  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer
  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
      The first of these two dimensions is n.
      Args:
        x: a Tensor with shape [..., m]
        n: an integer.
      Returns:
        a Tensor with shape [..., n, m/n]
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


def compute_qkv(q,
                m,
                total_key_depth,
                total_value_depth, ):
    """Computes query, key and value.
      Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
        total_key_depth: an integer
        total_value_depth: an integer
      Returns:
        q, k, v : [batch, length, depth] tensors
    """

    q = compute_attention_component(q, total_key_depth,
                                    "q")
    k = compute_attention_component(m, total_key_depth,
                                    "k")
    v = compute_attention_component(m, total_value_depth,
                                    "v")
    return q, k, v


def compute_attention_component(antecedent,
                                total_depth,
                                name="c"):
    """Computes attention compoenent (query, key or value).
    Args:
        antecedent: a Tensor with shape [batch, length, channels]
        total_depth: an integer
        filter_width: An integer specifying how wide you want the attention
          component to be.
        padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
        name: a string specifying scope name.
    Returns:
        c : [batch, length, depth] tensor
    """

    return tf.layers.dense(
        antecedent, total_depth, use_bias=False, name=name)
