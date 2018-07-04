"""Здесь определена функция потерь (пока три штуки: yandex, AT&T, triplet loss"""

import tensorflow as tf


def _pairwise_distances(labels, question, response, params, calculate_metric=False, top_K=1):
    """Compute the 2D matrix of cosine distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
    Returns:
        _pairwise_similarities: tensor of shape (batch_size, batch_size)
    """
    if params.loss_type == 'ATnT':
        dot_product = tf.matmul(question, tf.transpose(response))
        square_norm = tf.diag_part(dot_product)
        distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
        distances = tf.maximum(distances, 0.0)
    else:
        question_normalized = tf.nn.l2_normalize(question, axis=1)
        response_normalized = tf.nn.l2_normalize(response, axis=1)
        distances = tf.maximum(1 - tf.matmul(question_normalized, response_normalized, adjoint_b=True), 0.0)

    # TODO: считать P@K сразу для нескольких K
    if calculate_metric:
        K = tf.shape(distances)[0]   # сортируем всю выборку
        _, closest_indexes = tf.nn.top_k(-distances, k=K)  # сортируем
        closest_indexes_cropped = closest_indexes[:, 0:top_K]
        # reshaped_labels = tf.reshape(labels, [-1])
        print('closest_indexes_cropped', closest_indexes_cropped.shape)

        # PRECISION at K
        flatten_indexes = tf.reshape(closest_indexes_cropped, [1, -1])
        closest_labels = tf.reshape(tf.gather(labels, flatten_indexes), [-1, top_K])
        true_labels = tf.transpose(tf.reshape(tf.tile(labels, [top_K]), [top_K, -1]))
        num_equal = tf.reduce_sum(tf.cast(tf.equal(true_labels, closest_labels), tf.int32))
        precision_at_K = tf.divide(num_equal, tf.multiply(tf.shape(labels)[0], top_K))

        # MRR
        # counted = tf.bincount(tf.cast(labels, tf.int32))
        # how_many = tf.gather(counted, labels)  # для каждого примера считаем, сколько есть положительных примеров
        # mask = tf.cast(tf.subtract(how_many, 1), tf.bool)  # есть ли хотя бы один положительный пример
        #
        # closest_masked = tf.boolean_mask(closest_indexes[:, 1:], mask, axis=0)  # самые близкие интенты кроме себя
        # labels_masked = tf.boolean_mask(labels, mask, axis=0)
        #
        # reshaped_closest = tf.reshape(closest_masked, [1, -1])
        # selected_labels = tf.reshape(tf.gather(labels, reshaped_closest), [-1, K])  # все лейблы близких интентов
        #
        # labels_ = tf.transpose(tf.reshape(tf.tile(labels_masked, [K]), [K, -1]))
        # equal_labels = tf.to_float(tf.equal(labels_, selected_labels))
        #
        # indexes_of_first_true = tf.argmax(equal_labels, axis=1)
        # MRR = tf.reduce_mean(tf.divide(1, tf.add(indexes_of_first_true, 1)))
        MRR = 0

        return distances, precision_at_K, MRR
    else:
        return distances


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    # distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
    distinct_indices = tf.logical_and(j_not_equal_k, i_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def get_valid(loss_, mask_):
    """get num positive triplets and fraction
    Args:
        loss_: triplet loss
        mask_: valid triplets loss
    Returns:
        num_positive_triplets: num of pos triplets
        fraction_positive_triplets: fraction of pos triplets
    """
    valid_triplets = tf.to_float(tf.greater(loss_, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)

    num_valid_triplets = tf.reduce_sum(mask_)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    return num_positive_triplets, fraction_positive_triplets


def get_loss(labels, embeddings, params):
    """Build the triplet loss over a batch of embeddings.
    We generate all the valid triplets and average the loss over the positive ones.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        params: params for triplet loss
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    print('helllllllllll')
    if params.loss_type == 'usual':

        one_hot_labels = tf.one_hot(labels, 2)

        loss = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=embeddings)
            )
        return loss
    else:
        print('qwrq wrqasf')
        # Get the pairwise distance matrix
        question, response = embeddings
        question = tf.cast(question, dtype=tf.float32)
        response = tf.cast(response, dtype=tf.float32)

        pairwise_dist, p_at_k, mrr = _pairwise_distances(labels, question, response, params, True)
        # shape (batch_size, batch_size, 1)
        anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
        assert anchor_positive_dist.shape[2] == 1, "{}".format(anchor_positive_dist.shape)
        # shape (batch_size, 1, batch_size)
        anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)
        assert anchor_negative_dist.shape[1] == 1, "{}".format(anchor_negative_dist.shape)

        if params.loss_type == 'yandex':
            beta = params.beta
            margin = params.margin
            triplet_loss1 = anchor_positive_dist - beta + margin
            triplet_loss2 = beta - anchor_negative_dist + margin

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss1 = tf.maximum(triplet_loss1, 0.0)
            triplet_loss2 = tf.maximum(triplet_loss2, 0.0)
            triplet_loss = triplet_loss1 + triplet_loss2

            mask = _get_triplet_mask(labels)
            mask = tf.to_float(mask)
            triplet_loss = tf.multiply(mask, triplet_loss)

            # Count number of positive triplets (where triplet_loss > 0)
            num_positive_triplets, fraction_positive_triplets = get_valid(triplet_loss, mask)

            # Get final mean triplet loss over the positive valid triplets
            triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

            return triplet_loss, fraction_positive_triplets, p_at_k, mrr

        if params.loss_type == 'triplet':
            margin = params.margin
            triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

            mask = _get_triplet_mask(labels)
            mask = tf.to_float(mask)
            triplet_loss = tf.multiply(mask, triplet_loss)

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = tf.maximum(triplet_loss, 0.0)

            # Count number of positive triplets (where triplet_loss > 0)
            num_positive_triplets, fraction_positive_triplets = get_valid(triplet_loss, mask)

            # Get final mean triplet loss over the positive valid triplets
            triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

            return triplet_loss, fraction_positive_triplets, p_at_k, mrr

        if params.loss_type == 'ATnT':
            beta = params.beta
            N = params.N
            epsilon = params.epsilon

            anchor_positive_dist = -tf.log(-tf.divide(anchor_positive_dist, beta) + 1 + epsilon)
            anchor_negative_dist = -tf.log(-tf.divide((N - anchor_negative_dist), beta) + 1 + epsilon)

            triplet_loss = anchor_positive_dist + anchor_negative_dist

            mask = _get_triplet_mask(labels)
            mask = tf.to_float(mask)
            triplet_loss = tf.multiply(mask, triplet_loss)

            num_positive_triplets, fraction_positive_triplets = get_valid(triplet_loss, mask)

            triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

            return triplet_loss, fraction_positive_triplets, p_at_k, mrr
