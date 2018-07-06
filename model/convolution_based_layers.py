import tensorflow as tf
from .attention_module import shape_list, apply_norm, top_interactions, chose_rows_given_indices


def DCNN_layer(out, num_filters, kernel_size,
               kmax, bias_activation=tf.nn.relu,
               conv_activation=None, debug=False):
    """
    apply convolution row-wise (kernels differ for each row)(described in the http://www.aclweb.org/anthology/P14-1062)
      Args:
        out: A Tensor to be transformed. (bath, seq_len, filters)
        num_filters: an integer (filter size in each convolution)
        kernel_size: an integer (kernel size of convolutions)
        kmax: an integer (number of maximum values to return for each filter)
        conv_activation: an activation to apply after convolution
        bias_activation: an activation to apply after adding bias to each filter
        debug: whether to print some tensor shape
      Returns:
        a Tensor (bath, kmax, filters)
    """
    with tf.name_scope("dcnn_layer"):
        r_conv = row_convolution(out, num_filters=num_filters, kernel_size=kernel_size,
                                 activation=conv_activation)
        if debug: print('r_conv', r_conv.shape)
        pool = kmax_pooling(r_conv, kmax=kmax, sort=False)
        if debug: print('pool', pool.shape)
        with tf.variable_scope('add_bias_for_filter'):
            pool = tf.transpose(pool, perm=[0, 1, 3, 2])
            pool_b = tf.contrib.layers.bias_add(pool, activation_fn=bias_activation)
            pool_b = tf.transpose(pool_b, perm=[0, 1, 3, 2])
        if debug: print('pool_b', pool_b.shape)
    #         print(tf.trainable_variables())

    return pool_b


def folding(x, debug=False):
    """
    sums every two rows in a filters dimension component-wise. For a map of d rows, folding returns
    a map of d/2 rows (discribed in the http://www.aclweb.org/anthology/P14-1062)
      Args:
        x: A Tensor to be transformed. (bath, seq_len, filters)
        debug: bool
      Returns:
        a Tensor (bath, seq_len, filters/2)
    """
    input_unstack = tf.unstack(x, axis=2)
    out = []
    with tf.name_scope("folding"):
        for i in range(0, len(input_unstack), 2):
            fold = tf.add(input_unstack[i], input_unstack[i + 1])
            out.append(fold)
        fold = tf.stack(out, axis=2)
        if debug: print('fold', fold.shape)

    return fold


def row_convolution(x, num_filters, kernel_size, activation=None):
    """
    apply convolution row-wize (kernels differ for each row)(discribed in the http://www.aclweb.org/anthology/P14-1062)
      Args:
        x: A Tensor to be transformed. (bath, seq_len, filters)
        num_filters: an integer (filter size in each convolution)
        kernel_size: an integer (kernel size of convolutions)
        activation: an activation to apply after convolution
      Returns:
        a Tensor (bath, seq_len, filters) same shape as input
    """
    input_unstack = tf.unstack(x, axis=2)
    convs = []
    with tf.name_scope("row_conv"):
        for i, r in enumerate(input_unstack):
            with tf.variable_scope("rconv%d" % i):
                convs.append(tf.layers.conv1d(
                    inputs=r,
                    filters=num_filters,
                    kernel_size=kernel_size,
                    padding="same",
                    activation=activation,
                ))

        conv = tf.stack(convs, axis=2)
    return conv


def conv_kmaxpool_layer(out, num_filters, kernel_sizes, kmax, bias_activation=tf.nn.relu,
                        sort=False, conv_activation=None, debug=False):
    """
   Apply 1d convolutions and k max pooling the sequence.
      Args:
        out: A Tensor to be transformed. (bath, seq_len, filters)
        num_filters: an integer (filter size in each convolution)
        kernel_sizes: a list of integers (kernel size of convolutions)
        kmax: an integer (number of maximum values to return for each filter)
        sort: bool
        bias_activation: activation tf
        conv_activation: activation tf
        debug: bool
      Returns:
        a Tensor (bath, kmax, filters)
    """
    conv = words_convolution(out, num_filters=num_filters, kernel_sizes=kernel_sizes,
                             activation=conv_activation)
    if debug: print('conv', conv.shape)
    # conv = tf.layers.batch_normalization(conv)
    if conv_activation:
        conv = conv_activation(conv)
    pool = kmax_pooling(conv, kmax=kmax, sort=sort)
    if debug: print('pool', pool.shape)

    with tf.variable_scope('add_bias_for_filter'):
        pool_b = tf.contrib.layers.bias_add(pool, activation_fn=bias_activation)
    if debug: print('pool_b', pool_b.shape)

    return pool_b


def conv_kmaxpool_layer_first(out, num_filters, kernel_sizes, kmax, bias_activation=tf.nn.relu,
                              sort=False, conv_activation=None, debug=False):
    """
   Apply 1d convolutions and k max pooling the sequence.
      Args:
        out: A Tensor to be transformed. (bath, seq_len, filters)
        num_filters: an integer (filter size in each convolution)
        kernel_sizes: a list of integers (kernel size of convolutions)
        kmax: an integer (number of maximum values to return for each filter)
        sort: bool
        bias_activation: activation tf
        conv_activation: activation tf
        debug: bool
      Returns:
        a Tensor (bath, kmax, filters)
    """
    conv = words_convolution(out, num_filters=num_filters, kernel_sizes=kernel_sizes,
                             activation=conv_activation)
    if debug: print('conv', conv.shape)
    conv = tf.layers.batch_normalization(conv)
    if conv_activation:
        conv = conv_activation(conv)
    pool = top_interactions(conv, kmax=kmax)
    if debug: print('pool', pool.shape)

    with tf.variable_scope('add_bias_for_filter'):
        pool_b = tf.contrib.layers.bias_add(pool, activation_fn=bias_activation)
    if debug: print('pool_b', pool_b.shape)

    return pool_b


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


def words_convolution(out, num_filters, kernel_sizes, activation=None):
    """
    Apply 1d convolutions with different kernel sizes to the sequence.
      Args:
        out: A Tensor to be transformed. (bath, seq_len, filters)
        num_filters: an integer (filter size in each convolution)
        kernel_sizes: a list of integers (kernel size of convolutions)
        activation: an activation
      Returns:
        a Tensor (bath, num_filters*len(kernel_sizes), filters)
    """
    conv_list = []
    with tf.name_scope('conv'):
        for kernel_size in kernel_sizes:
            with tf.variable_scope("conv1d%d" % kernel_size):
                conv_list.append(tf.layers.conv1d(
                    inputs=out,
                    filters=num_filters,
                    kernel_size=kernel_size,
                    padding="same",
                    activation=activation
                ))
        conv = tf.concat(conv_list, axis=-1, name='concat')
    return conv