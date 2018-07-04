"""kNN model"""

import nmslib
import numpy as np
from ml_utils import pklgz


class NearestNeighbors(object):
    def __init__(self, method_name='hnsw', space_name='cosinesimil', dim=300, n_neighbors=1, ef_construction=200, M=15,
                 post=0, skip_optimized_index=0, ef_search=50):
        self.vector_dim = dim
        self.n_neighbors = n_neighbors
        self.index_params = {'M': M, 'efConstruction': ef_construction, 'post': post,
                             'skip_optimized_index': skip_optimized_index}
        self.space_name = space_name
        self.method_name = method_name
        self.query_time_params = {'efSearch': ef_search}

    def fit(self, X_train):
        print('Index vector dim:', self.vector_dim)
        print('Index space name:', self.space_name)
        self.index = nmslib.init(method=self.method_name, space=self.space_name)
        self.index.addDataPointBatch(X_train)
        print('Index params:', self.index_params)
        self.index.createIndex(self.index_params, print_progress=True)
        print('Index created.')
        print('Setting ef search parameter:', self.query_time_params)
        self.index.setQueryTimeParams(self.query_time_params)

        return self

    def kneighbors(self, X, return_distances=False):
        # X - array with shape=(vector_dim,) k=self.n_neighbors - number of closest elements (returns 2 numpy arrays)
        if return_distances:
            return self.index.knnQuery(X, k=self.n_neighbors)
        else:
            return self.index.knnQuery(X, k=self.n_neighbors)[0]

    def kneighbors_batch(self, X, return_distances=False):
        # Query of X, k=self.n_neighbors - number of closest elements (returns 2 numpy arrays)
        neigh_and_dist = self.index.knnQueryBatch(X, k=self.n_neighbors)
        if return_distances:
            return np.array(neigh_and_dist)[:, 0, :].astype(np.int32), np.array(neigh_and_dist)[:, 1, :]
        else:
            return np.array(neigh_and_dist)[:, 0, :].astype(np.int32)

    def save_index(self, index_path):
        self.index.saveIndex(index_path)

    def load_index(self, filepath):
        print('Index vector dim:', self.vector_dim)
        print('Index space name:', self.space_name)
        self.index = nmslib.init(method=self.method_name, space=self.space_name)
        self.index.loadIndex(filepath)
        print('Index params:', self.index_params)
        print('Setting ef search parameter:', self.query_time_params)
        self.index.setQueryTimeParams(self.query_time_params)
        print('Index loaded.')


class KNeighborsClassifier(NearestNeighbors):
    def __init__(self, method_name='hnsw', space_name='cosinesimil', dim=300, n_neighbors=1, ef_construction=200, M=15,
                 post=0, skip_optimized_index=0, ef_search=50):
        super().__init__(method_name, space_name, dim, n_neighbors, ef_construction, M, post, skip_optimized_index,
                         ef_search)

    def fit(self, X_train, y_train):
        self.y_train = y_train
        super().fit(X_train)

        return self

    # TODO: Прикрутить вызов функции к предикту
    def predict(self, X_test, f=None):
        """
        Intended for one element, not list or array.
        f(labels, distances) - callable function, returns a predicted label of X_test.
        If f is None, return label of nearest neighbour.
        """
        indicies, distances = super().kneighbors(X_test, return_distances=True)
        if f is not None:
            return f(self.y_train[indicies], distances)
        else:
            return self.y_train[indicies][:, 0]

    def get_labels_and_distances(self, X_test):
        indicies, distances = super().kneighbors_batch(X_test, return_distances=True)
        return indicies, distances

    def predict_batch(self, X_test, f=None):
        """
        Intended for array
        f(labels, distances) - callable function, returns array of predicted labels with shape (len(X_test),).
        If f is None, return label of nearest neighbour for each element of X_test.
        """
        indicies, distances = super().kneighbors_batch(X_test, return_distances=True)
        if f is not None:
            return f(self.y_train[indicies], distances)
        else:
            return self.y_train[indicies][:, 0]

    def save_model(self, index_path, y_train_path):
        super().save_index(index_path)
        pklgz.dump(self.y_train, y_train_path)

    def load_model(self, index_path, y_train_path):
        super().load_index(index_path)
        self.y_train = pklgz.load(y_train_path)