import sys
from collections import Counter
from enum import Enum
import numpy as np
import pyprind
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier


class DistType(Enum):
    Euclidian = "euclidean"
    OneClass = "one_class"


BLACK = 0


class DistanceCalculator:
    def __init__(self, batch_size=1):
        self._batch_size = batch_size

    def euclidean(self, mx1, mx2, ignore, typ="euclidean"):
        # Get euclidean distances as 2D array
        dists = cdist(mx1, mx2, typ)
        # return the most distant rows
        # mx1 - test matrix
        # mx2 - train matrix
        # index of max value in dists (r, c)
        # TODO change to average distance ?
        # TODO RETURN K
        # TODO Think about first reveal
        top_index = dists.mean(axis=1).argsort(kind='heapsort').tolist()
        return [i for i in top_index if i not in ignore][0:self._batch_size]


class Learning:
    def __init__(self, batch_size=1):
        self._batch_size = batch_size

    def machine_learning(self, x_train, y_train, x_test, smallest_class, ignore, clf=None):
        # smallest class -> 1/0
        if clf is None:
            # n_estimators - number of trees
            # balances -
            clf = RandomForestClassifier(n_estimators=200, class_weight="balanced")
        clf.fit(np.asmatrix(x_train, dtype=np.float32), y_train)
        probs = clf.predict_proba(x_test)
        # probs closer to 0 -> 0
        # sorting high to low hence the minus sign
        top_index = (-probs[:, smallest_class]).argsort(kind='heapsort').tolist()  # smallest class black
        return [i for i in top_index if i not in ignore][0:self._batch_size]


class TimedActiveLearning:
    def __init__(self, params, num_black):
        self._init_variables(params)
        self._smallest_class = BLACK  # who is the black
        self._n_black = num_black  # Counter(labels)[BLACK]  # number of blacks
        self._stop_cond = np.round(self._target_recall * self._n_black)  # number of blacks to find - stop condition

    def _init_variables(self, params):
        self._dist_type = params['dist_type']
        self._batch_size = params['batch_size']
        self._eps = params['eps']
        self._target_recall = params['target_recall']

        self._tags = None

        # initialize the train objects with copies - COPY
        self._x_test_matrix = None
        self._y_test = None
        self._x_train_matrix = None
        self._x_train_idx = []
        self._y_train = []

        self._time = 0  # how many nodes we asked about
        self._num_black_found = 0
        self._first_time = True

    def _first_exploration(self):
        self._bar = pyprind.ProgBar(len(self._tags), stream=sys.stdout)  # optional
        # explore first using distance
        start_k = 2 if self._batch_size == 1 else 1
        for i in range(start_k):
            # first two nodes (index)
            top_index = DistanceCalculator(batch_size=self._batch_size).euclidean(self._x_test_matrix,
                                                                                  self._x_test_matrix, self._x_train_idx)
            self._reveal(top_index)

    def _explore_exploit(self):
        rand = np.random.uniform(0, 1)
        # 0 < a < eps -> distance based  || at least one black and white reviled -> one_class/ euclidean
        if rand < self._eps or len(Counter(self._y_train)) < 2:
            # idx -> most far away node index
            top_index = DistanceCalculator(batch_size=self._batch_size).euclidean(self._x_test_matrix,
                                                                                  self._x_train_matrix, self._x_train_idx)
        else:
            # idx -> by learning
            top_index = Learning(batch_size=self._batch_size).machine_learning(self._x_train_matrix, self._y_train,
                                                                               self._x_test_matrix, self._smallest_class
                                                                               , self._x_train_idx)
        self._reveal(top_index)

    def _reveal(self, top_index):
        for idx in top_index:
            if self._y_test[idx] == self._smallest_class:
                self._num_black_found += 1

            self._x_train_idx.append(idx)
            # add feature vec to train
            self._y_train.append(self._y_test[idx])
        end = 0

    def _forward_time_data(self, beta_matrix, labels):
        self._y_test = labels[:]
        self._x_test_matrix = beta_matrix
        self._x_train_matrix = np.vstack([self._x_test_matrix[idx, :] for idx in self._x_train_idx]) if self._x_train_idx else None
        self._tags = labels  # for display

    def step(self, beta_matrix, labels):
        self._forward_time_data(beta_matrix, labels)
        # first exploration - reveal at least two graph
        if self._first_time:
            self._first_time = False
            self._first_exploration()
        else:
            self._explore_exploit()
        print("recall: " + str(self._num_black_found / self._n_black))
        print(str(len(self._x_train_idx)) + " | " + str(len(self._tags)) + " graphs revealed -- " +
              str(len(self._x_train_idx) / len(self._tags)) + "%")
        return len(self._x_train_idx) / len(self._tags), self._num_black_found / self._n_black  # revealed, recall
