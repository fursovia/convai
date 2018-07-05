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


class serving_input_fn:
    def __init__(self):
        self.features = {'cont': tf.placeholder(tf.int64, shape=[None, 140]),
                         'quest': tf.placeholder(tf.int64, shape=[None, 140]),
                         'resp': tf.placeholder(tf.int64, shape=[None, 140]),
                         'facts': tf.placeholder(tf.int64, shape=[None, 5 * 140])}

        self.receiver_tensors = self.features
        self.receiver_tensors_alternatives = None


class pred_agent():
    def __init__(self, args, raw_utterances, train_embeddings_path, train_model):
        self.args = args
        self.estimator = self.create_model()
        self.create_predictor()
        self.raw_utterances = raw_utterances
        self.train_embeddings_path = train_embeddings_path
        self.fit_knn(train_model=train_model)

        vocabs_path = os.path.join(self.args.data_dir, 'vocabs')
        uni2idx_path = os.path.join(vocabs_path, 'uni2idx.pkl')
        bi2idx_path = os.path.join(vocabs_path, 'bi2idx.pkl')
        char2idx_path = os.path.join(vocabs_path, 'char2idx.pkl')

        self.uni2idx = pickle.load(open(uni2idx_path, 'rb'))
        self.bi2idx = pickle.load(open(bi2idx_path, 'rb'))
        self.char2idx = pickle.load(open(char2idx_path, 'rb'))

    def create_model(self):
        tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.INFO)

        # ПОДГРУЖАЕМ ПАРАМЕТРЫ
        json_path = os.path.join(self.args.model_dir, 'params.json')
        assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
        self.params = Params(json_path)

        # ОПРЕДЕЛЯЕМ МОДЕЛЬ
        tf.logging.info("Creating the model...")
        config = tf.estimator.RunConfig(tf_random_seed=230, model_dir=self.args.model_dir)

        estimator = tf.estimator.Estimator(model_fn, params=self.params, config=config)
        return estimator

    def create_predictor(self):
        self.predictor = tf.contrib.predictor.from_estimator(
            self.estimator,
            serving_input_fn
        )

    def fit_knn(self, train_model):
        train_embeddings = pickle.load(open(self.train_embeddings_path, 'rb'))
        if train_model:
            self.knn_model = KNeighborsClassifier(n_neighbors=5).fit(train_embeddings, np.zeros_like(train_embeddings))
            self.knn_model.save_index('model/knn.index')
        else:
            self.knn_model = KNeighborsClassifier
            self.knn_model.load_index('model/knn.index')

    def choose_from_knn(self, q_embeddings):
        indicies, _ = self.knn_model.get_labels_and_distances(q_embeddings)
        chosen = self.raw_utterances[indicies]
        print('from where to choose: ', chosen)
        return str(chosen[0][0])

    def predict(self, super_dict):
        vocabs = [self.uni2idx, self.bi2idx, self.char2idx]
        print('vocabs')
        data_to_predict_knn = inference_time(super_dict, np.zeros((1, 140)), vocabs, 1)
        # print('predict this data')
        test_predictions_knn = self.predictor({'cont': data_to_predict_knn[:, 0].reshape(-1, 140),
                                               'quest': data_to_predict_knn[:, 1].reshape(-1, 140),
                                               'resp': data_to_predict_knn[:, 2].reshape(-1, 140),
                                               'facts': data_to_predict_knn[:, 3:].reshape(-1, 5 * 140)})

        # print('test_predictions_knn', test_predictions_knn)

        qemb = []
        for p in test_predictions_knn['hist_emb']:
            #             print('p', p)
            qemb.append(p)
        qemb = np.array(qemb, float).reshape(-1, 300)

        chosen = self.choose_from_knn(qemb)
        print('second time: ', chosen)
        return chosen
