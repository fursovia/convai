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

class pred_agent():
    def __init__(self, args, super_dict, raw_utterances, vectorized_responses):
        self.args = args
        self.estimator = self.create_model()
        self.super_dict = super_dict
        self.raw_utterances = raw_utterances
        self.vectorized_responses = vectorized_responses


    def create_model(self):
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)

        # ПОДГРУЖАЕМ ПАРАМЕТРЫ
        json_path = os.path.join(self.args['model_dir'], 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        self.params = Params(json_path)

        # ОПРЕДЕЛЯЕМ МОДЕЛЬ
        tf.logging.info("Creating the model...")
        config = tf.estimator.RunConfig(tf_random_seed=230, model_dir=self.args['model_dir'])

        estimator = tf.estimator.Estimator(model_fn, params=self.params, config=config)
        return estimator

    def create_predictor(self):
        self.predictor = tf.contrib.predictor.from_estimator(
            self.estimator,
            serving_input_fn
        )

    def predict(self):

        vocabs_path = os.path.join(self.args.data_dir, 'vocabs')
        uni2idx_path = os.path.join(vocabs_path, 'uni2idx.pkl')
        bi2idx_path = os.path.join(vocabs_path, 'bi2idx.pkl')
        char2idx_path = os.path.join(vocabs_path, 'char2idx.pkl')

        uni2idx = pickle.load(open(uni2idx_path, 'rb'))
        bi2idx = pickle.load(open(bi2idx_path, 'rb'))
        char2idx = pickle.load(open(char2idx_path, 'rb'))

        vocabs = [uni2idx, bi2idx, char2idx]

        data_to_predict = inference_time(self.super_dict, self.vectorized_responses, vocabs)

        test_predictions = self.predictor({'text': data_to_predict})  #['y_prob']

        preds = []
        for i, p in enumerate(test_predictions):
            preds.append(p['y_prob'])

        preds = np.array(preds, float)
        max_element = np.argmax(preds)

        what_to_return = self.raw_utterances[max_element]

        return what_to_return
