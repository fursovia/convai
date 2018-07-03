"""Позволяет перевести данные в эмбединги с помощью предобученной модели"""

import tensorflow as tf
import argparse
import os
from model.utils import Params
from model.input_fn import input_fn
from model.model_fn import model_fn
import pickle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments')
parser.add_argument('--data_dir', default='/data/i.fursov/data')


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parser.parse_args()

    json_path = os.path.join(args.model_dir, 'params.json')
    params = Params(json_path)

    # ОПРЕДЕЛЯЕМ МОДЕЛЬ
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # ПОЛУЧАЕМ ВСЕ ЭМБЕДИНГИ ИЗ ТРЕЙНА
    tf.logging.info("Predicting train data...")
    train_predictions = estimator.predict(lambda: input_fn(args.data_dir, params, 'full', False))

    train_embeddings = np.empty((0, 300))
    for i, p in enumerate(train_predictions):
        train_embeddings = np.append(train_embeddings, p['resp_emb'].reshape(-1, 300), axis=0)

    train_emb_path = os.path.join(args.model_dir, 'embeddings.pkl')
    pickle.dump(train_embeddings, open(train_emb_path, 'wb'))
