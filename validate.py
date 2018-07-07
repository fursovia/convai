import tensorflow as tf
import tensorflow as tf
import os
from model.utils import Params
from model.model_fn import model_fn
import pickle
import numpy as np
from model.utils import inference_time
from knn import KNeighborsClassifier, NearestNeighbors
import tensorflow as tf
import os
from model.utils import Params
from model.model_fn import model_fn
import pickle
import numpy as np
from model.utils import inference_time
import argparse
import os
import pickle
import pandas as pd
from tg_prediction import pred_agent
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/data/i.fursov/convai/last_exp')
parser.add_argument('--data_dir', default='/data/i.anokhin/convai/new_datatrain')
parser.add_argument('--model_dir', default='experiments')
parser.add_argument('--data_dir', default='/data/i.anokhin/convai/data/only1')
parser.add_argument('--train_knn', default='Y')
parser.add_argument('--emb_dim', type=int, default=300)


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    raw_utts = pickle.load(open(os.path.join(args.data_dir, 'raw_responses.pkl'), 'rb'))
    emb_path = os.path.join(args.model_dir, 'embeddings.pkl')

    if args.train_knn == 'Y':
        train = True
    else:
        train = False

    agent = pred_agent(args, raw_utts, emb_path, train, args.emb_dim)

    super_dict = {'context': [],
                  'question': ''}
                  #'facts': []}

    test_data = pd.read_csv('/data/i.fursov/convai/data/only1/full.csv')
    test_data = test_data.fillna('')
    indexes = list(test_data.index)
    rand = np.random.choice(indexes, 1)[0]
    random_facts = test_data.loc[rand, ['fact1', 'fact2', 'fact3', 'fact4', 'fact5']].tolist()
    super_dict['facts'] = random_facts

    while True:
        if len(super_dict['context']) > 3:
            super_dict['context'] = super_dict['context'][-3:]
        print('cont len = {}'.format(len(super_dict['context'])))

        text = input('input message: \n')
        super_dict['question'] = str(text.strip())
        print('facts = {}'.format(super_dict['facts']))
        print('context = {}'.format(super_dict['context']))
        print('question = {}'.format(super_dict['question']))
        ans = agent.predict(super_dict)
        print('answer = {} \n'.format(ans))

        super_dict['context'].append((text,))
        super_dict['context'].append((ans,))


