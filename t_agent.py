import tensorflow as tf
import argparse
import os
from model.utils import Params
from model.model_fn import model_fn
import pickle
import numpy as np
import copy
from model.utils import text2vec
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from tensorflow.python.client import device_lib


class serving_input_fn:
    def __init__(self):
        self.features = tf.placeholder(tf.int64, shape=[None, 200])
        self.receiver_tensors = {
           'text': self.features,
        }
        self.receiver_tensors_alternatives = None


class DSSMAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        DictionaryAgent.add_cmdline_args(argparser)

        agent = argparser.add_argument_group('DSSM Arguments')
        agent.add_argument('--model_dir', default='experiments',
                    help="Experiment directory containing params.json")
        agent.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")

    def __init__(self, opt):
        super().__init__(opt)
        self.id = 'DSSM'
        self.observation = {}
        self.episode_done = True
        self.estimator = self.create_model()
        self.opt = opt
        self.create_predictor()

    def txt2vec(self, txt):
        return np.array(self.dict.txt2vec(txt)).astype(np.int32)

    def vec2txt(self, vec):
        return self.dict.vec2txt(vec)

    def observe1(self, observation):
        observation = copy.deepcopy(observation)

        if not self.episode_done:
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']

        self.observation = observation
        self.episode_done = observation['episode_done']

        return observation

    def observe(self, observation):
        """Save observation for act.
        If multiple observations are from the same episode, concatenate them.
        """
        # shallow copy observation (deep copy can be expensive)
        obs = observation.copy()
        batch_idx = self.opt.get('batchindex', 0)
        self.observation = obs
        #self.answers[batch_idx] = None
        return obs


    def create_model(self):

        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)

        # ПОДГРУЖАЕМ ПАРАМЕТРЫ
        json_path = os.path.join(self.opt['model_dir'], 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        self.params = Params(json_path)

        # ОПРЕДЕЛЯЕМ МОДЕЛЬ
        tf.logging.info("Creating the model...")
        config = tf.estimator.RunConfig(tf_random_seed=230, model_dir=self.opt['model_dir'])

        estimator = tf.estimator.Estimator(model_fn, params=self.params, config=config)

        return estimator


    def create_predictor(self):
        self.predictor = tf.contrib.predictor.from_estimator(
            self.estimator,
            serving_input_fn
        )


    def predict(self, some_dicts):
        word2idx = pickle.load(open(os.path.join(self.opt['data_dir'], 'word2idx.pkl'), 'rb'))

        data_to_predict = []
        candidates = []

        for dict_ in some_dicts:
            test_data, true_id, true_ans, raw_dial, cands = text2vec(dict_, word2idx)
            data_to_predict.append(test_data)
            candidates.append(cands)

        data_to_predict = np.array(data_to_predict, int).reshape(-1, 200)

        # подаем по 20 кандидатов и находим лучшие из них
        # test_input_fn = tf.estimator.inputs.numpy_input_fn(data_to_predict,
        #                                                    num_epochs=1,
        #                                                    batch_size=20,
        #                                                    shuffle=False)

        preds = self.predictor({'text': data_to_predict})
        print(preds)
        predis = preds['predict'].reshape(-1, 20)

        # test_predictions = self.estimator.predict(test_input_fn,
        #                                           yield_single_examples=False)

        output = []
        for i, batch in enumerate(predis):
            sorted_elements = np.argsort(batch['y_prob'])[::-1]
            cands = np.array(candidates[i], object)
            ppp = cands[sorted_elements]
            output.append(ppp)

        return output


    def batchify(self, obs):
        """Convert batch observations `text` and `label` to rank 2 tensor `xs` and `ys`
        """
        def txt2np(txt, use_offset=True):
            vec = [self.txt2vec(t) for t in txt]
            max_len = max([len(v) for v in vec])
            arr = np.zeros((len(vec), max_len)).astype(np.int32) # 0 filled rank 2 tensor
            for i, v in enumerate(vec):
                offset = 0
                if use_offset:
                    offset = max_len - len(v) # Right justified
                for j, idx in enumerate(v):
                    arr[i][j + offset] = idx
            return arr # batch x time

        exs = [ex for ex in obs if 'text' in ex]
        valid_inds = [i for i, ex in enumerate(obs) if 'text' in ex]

        if len(exs) == 0:
            return (None,)*3

        xs = [ex['text'] for ex in exs]
        xs = txt2np(xs)
        ys = None
        if 'labels' in exs[0]:
            ys = [' '.join(ex['labels']) for ex in exs]
            ys = txt2np(ys, use_offset=False)
        return xs, ys, valid_inds

    def batch_act(self, observations):
        # observations:
        #       [{'label_candidates': {'office', ...},
        #       'episode_done': False, 'text': 'Daniel ... \nWhere is Mary?',
        #       'labels': ('office',), 'id': 'babi:Task10k:1'}, ...]

        batchsize = len(observations)
        batch_reply = [{'id': self.id} for _ in range(batchsize)]

        predictions = self.predict(observations)

        for i in range(len(batch_reply)):
            batch_reply[i]['text_candidates'] = predictions[i]
            batch_reply[i]['text'] = batch_reply[i]['text_candidates'][0]

        return batch_reply # [{'text': 'bedroom', 'id': 'RNN'}, ...]

    def act(self):
        return self.batch_act([self.observation])[0]
