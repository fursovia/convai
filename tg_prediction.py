import tensorflow as tf
import argparse
import os
from model.utils import Params
from model.model_fn import model_fn
import pickle
import numpy as np
from model.input_fn import pred_input_fn
from model.utils import inference_time
import pandas as pd
from model.input_fn import pred_input_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments')
parser.add_argument('--data_dir', default='data')


class serving_input_fn:
    def __init__(self):
        self.features = tf.placeholder(tf.int64, shape=[None, 140])
        self.receiver_tensors = {
           'text': self.features,
        }
        self.receiver_tensors_alternatives = None


if __name__ == '__main__':

    args = parser.parse_args()

    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    data = pd.read_csv(os.path.join(args.data_dir, 'initial/full.csv'))
    data2 = pd.read_csv(os.path.join(args.data_dir, 'raw_df.csv'))
    uts2 = data2['reply'].values
    all_utts = data['reply'].values
    unique_utts, indexes = np.unique(all_utts, return_index=True)
    raw_utts = uts2[indexes]

    # ПОДГРУЖАЕМ ПАРАМЕТРЫ
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # ОПРЕДЕЛЯЕМ МОДЕЛЬ
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230, model_dir=args.model_dir)

    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)


    # data_to_predict = inference_time(DICT_FROM_MISHA, unique_utts)  # dataframe

    train_predictions = estimator.predict(pred_input_fn(data_to_predict))

    preds = []
    for i, p in enumerate(train_predictions):
        preds.append(p['y_prob'])

    preds = np.array(preds, float)
    max_el = np.argmax(preds)

    what_to_return = raw_utts[max_el]