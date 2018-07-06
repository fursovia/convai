"""Позволяет перевести данные в эмбединги с помощью предобученной модели"""

import tensorflow as tf
import argparse
import os
from model.utils import Params
from model.input_fn import input_fn
from model.model_fn import model_fn
import pickle
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='exp')
parser.add_argument('--data_dir', default='/data/i.fursov/convai/data/only1')
parser.add_argument('--emb_dim', type=int, default=300)


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
    train_predictions = estimator.predict(lambda: input_fn(args.data_dir, params, 'unique_data', False))

    # train_embeddings = np.empty((0, 300))
    # for p in tqdm(train_predictions):
    #     train_embeddings = np.append(train_embeddings, p['resp_emb'].reshape(-1, 300), axis=0)

    train_embeddings = []
    for i, p in tqdm(enumerate(train_predictions)):
        train_embeddings.append(p['resp_emb'])
        # if i > 100:
        #     break
    train_embeddings = np.array(train_embeddings, float).reshape(-1, args.emb_dim)

    train_emb_path = os.path.join(args.model_dir, 'embeddings.pkl')
    pickle.dump(train_embeddings, open(train_emb_path, 'wb'))

    print('saved at {}'.format(train_emb_path))
