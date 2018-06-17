import tensorflow as tf
import argparse
import os
from model.utils import Params
from model.model_fn import model_fn
import pickle
import numpy as np

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
        # don't enter this loop for shared instantiations


        self.id = 'DSSM'
#             self.dict = DictionaryAgent(opt)
        self.observation = {}

#             self.path = opt.get('model_file', None)

#         if opt.get('model_dir') and os.path.isdir(opt['model_dir']):
#             print('Loading existing model parameters from ' + opt['model_dir'])
#             self.estimator = self.create_model()
#         else:
#             self.estimator = None

        self.episode_done = True


    def txt2vec(self, txt):
        return np.array(self.dict.txt2vec(txt)).astype(np.int32)

    def vec2txt(self, vec):
        return self.dict.vec2txt(vec)

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        # At this moment `self.episode_done` is the previous example
        if not self.episode_done:
            # If the previous example is not the end of the episode,
            # we need to recall the `text` mentioned in the previous example.
            # At this moment `self.observation` is the previous example.
            prev_dialogue = self.observation['text']
            # Add the previous and current `text` and update current `text`
            observation['text'] = prev_dialogue + '\n' + observation['text']
        # Overwrite with current example
        self.observation = observation
        # The last example of an episode is provided as `{'episode_done': True}`
        self.episode_done = observation['episode_done']
        return observation

    def create_model(self):

        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)

        # ПОДГРУЖАЕМ ПАРАМЕТРЫ
        json_path = os.path.join(agent.model_dir, 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        self.params = Params(json_path)

        # ОПРЕДЕЛЯЕМ МОДЕЛЬ
        tf.logging.info("Creating the model...")
        config = tf.estimator.RunConfig(tf_random_seed=230,
                                        model_dir=agent.model_dir,
                                        save_summary_steps=params.save_summary_steps)
        estimator = tf.estimator.Estimator(model_fn, params=self.params, config=config)

        return estimator

    def predict(self, test_data):
#         test_data = pickle.load(open(os.path.join(args.data_dir, 'test/X.pkl'), 'rb'))  # already stacked (C, Q, R, I)
        # raw_test_data = pickle.load(open(os.path.join(args.data_dir, 'test/R_raw.pkl'), 'rb'))  # raw replies

        # подаем по 10 кандидатов и находим лучшие из них
        test_input_fn = tf.estimator.inputs.numpy_input_fn(test_data,
                                                           num_epochs=1,
                                                           batch_size=10,
                                                           shuffle=False)

        test_predictions = self.estimator.predict(test_input_fn,
                                         predict_keys=['y_prob'],
                                         yield_single_examples=False)

        for i, batch in enumerate(test_predictions):
            best_id = 10 * i + np.argmax(batch['y_prob'])
            sorted_elements = np.argsort(batch['y_prob'], axis=1)
            print('Most likely answer id = {}'.format(best_id))
            print('Sorted elements: {}'.format(sorted_elements))

        pred = sorted_elements

            # print('Best raw reply: {}'.format(raw_test_data[best_id]))
#         idx = self.sess.run(self.idx, feed_dict={self.xs: xs, self.drop: False})
#         preds = [self.vec2txt([i]) for i in idx]
#         if random.random() < 0.1:
#             print('prediction:', preds[0])

        return preds

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
        for ex in batch_reply:
            ex['text_candidates'] = observations['label_candidates']


        return batch_reply # [{'text': 'bedroom', 'id': 'RNN'}, ...]

    def act(self):
        return self.batch_act([self.observation])[0]
