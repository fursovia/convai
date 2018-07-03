import tensorflow as tf
import os
from model.utils import Params
from model.model_fn import model_fn
import pickle
import numpy as np
from model.utils import inference_time


class serving_input_fn:
    def __init__(self):
        self.features = tf.placeholder(tf.int64, shape=[None, 140])
        self.receiver_tensors = {
           'text': self.features,
        }
        self.receiver_tensors_alternatives = None

class pred_agent():
    def __init__(self, args, raw_utterances, vectorized_responses, embeded_responses):
        self.args = args
        self.estimator = self.create_model()
        self.raw_utterances = raw_utterances
        self.vectorized_responses = vectorized_responses
        self.embeded_responses = embeded_responses


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

    def choose_from_knn(self, q_embeddings):
        _, indexes = None  # knn(q_embeddings)
        chosen_vectorized_elements = self.vectorized_responses[indexes]
        chosen_raw_elemenst = self.raw_utterances[indexes]

        return chosen_vectorized_elements, chosen_raw_elemenst

    def predict(self, super_dict):

        vocabs_path = os.path.join(self.args.data_dir, 'vocabs')
        uni2idx_path = os.path.join(vocabs_path, 'uni2idx.pkl')
        bi2idx_path = os.path.join(vocabs_path, 'bi2idx.pkl')
        char2idx_path = os.path.join(vocabs_path, 'char2idx.pkl')

        uni2idx = pickle.load(open(uni2idx_path, 'rb'))
        bi2idx = pickle.load(open(bi2idx_path, 'rb'))
        char2idx = pickle.load(open(char2idx_path, 'rb'))

        vocabs = [uni2idx, bi2idx, char2idx]

        data_to_predict_knn = inference_time(super_dict, self.vectorized_responses, vocabs, 1)  # ['q_emb']
        test_predictions_knn = self.predictor({'text': data_to_predict_knn})

        qemb = []
        for i, p in enumerate(test_predictions_knn):
            qemb.append(p['q_emb'])
        qemb = np.array(qemb, float).reshape(-1, 300)

        chosen_vects, chosen_raw = self.choose_from_knn(qemb)

        data_to_predict = inference_time(super_dict, chosen_vects, vocabs)

        test_predictions = self.predictor({'text': data_to_predict})  # ['y_prob']

        preds = []
        for i, p in enumerate(test_predictions):
            preds.append(p['y_prob'])

        preds = np.array(preds, float)
        max_element = np.argmax(preds)

        what_to_return = chosen_raw[max_element]

        return what_to_return
