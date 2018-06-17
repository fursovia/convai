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
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=self.opt['model_dir'])

        estimator = tf.estimator.Estimator(model_fn, params=self.params, config=config)

        return estimator

    def predict(self, some_dict):
        word2idx = pickle.load(open(os.path.join(self.opt['data_dir'], 'word2idx.pkl'), 'rb'))

        test_data, true_id, true_ans, raw_dial, cands = text2vec(some_dict, word2idx)

        # подаем по 20 кандидатов и находим лучшие из них
        test_input_fn = tf.estimator.inputs.numpy_input_fn(test_data,
                                                           num_epochs=1,
                                                           batch_size=20,
                                                           shuffle=False)

        test_predictions = self.estimator.predict(test_input_fn,
                                                  predict_keys=['y_prob'],
                                                  yield_single_examples=False)

        sorted_elements = np.argsort(test_predictions['y_prob'])[::-1]
        cands = np.array(cands, object)

        return cands[sorted_elements]


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

#         xs, ys, valid_inds = self.batchify(observations)

        # if xs is None:
        #     return batch_reply
        # else:
        #     preds = self.predict(xs)

#         for i in range(len(preds)):
#             batch_reply[valid_inds[i]]['text'] = preds[i]
        for i in range(len(batch_reply)):
            # print(ob)
            batch_reply[i]['text_candidates'] = list(observations[i]['label_candidates'])
            batch_reply[i]['text'] = batch_reply[i]['text_candidates'][0]

        return batch_reply # [{'text': 'bedroom', 'id': 'RNN'}, ...]

    def act(self):
        return self.batch_act([self.observation])[0]
