"""predict data"""

import tensorflow as tf
import argparse
import os
from model.utils import Params
from model.model_fn import model_fn
import pickle
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # ПОДГРУЖАЕМ ПАРАМЕТРЫ
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # ОПРЕДЕЛЯЕМ МОДЕЛЬ
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # TODO: здесь прикрутить функцию, которая будет из сырого текста делать подходящий формат
    test_data = pickle.load(open(os.path.join(args.data_dir, 'test/X.pkl'), 'rb'))  # already stacked (C, Q, R, I)
    # raw_test_data = pickle.load(open(os.path.join(args.data_dir, 'test/R_raw.pkl'), 'rb'))  # raw replies

    # подаем по 10 кандидатов и находим лучшие из них
    test_input_fn = tf.estimator.inputs.numpy_input_fn(test_data,
                                                       num_epochs=1,
                                                       batch_size=10,
                                                       shuffle=False)

    test_predictions = estimator.predict(test_input_fn,
                                         predict_keys=['y_prob'],
                                         yield_single_examples=False)

    for i, batch in enumerate(test_predictions):
        best_id = 10 * i + np.argmax(batch['y_prob'])
        print('Most likely answer id = {}'.format(best_id))
        # print('Best raw reply: {}'.format(raw_test_data[best_id]))
