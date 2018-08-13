from sklearn import linear_model

from beta_calculator import BetaCalculator
from graphs_al import Graphs
from loggers import BaseLogger, PrintLogger
import numpy as np


class AllBetaPairs(BetaCalculator):
    def __init__(self, graphs: Graphs, split=1):
        self._interval = int(graphs.number_of_graphs() / split)
        self._all_features = []
        for graph in graphs.graph_names():
            m = graphs.features_matrix(graph)
            # self._nodes_for_graph.append(m.shape[0])
            # append graph features
            self._all_features.append(m)
            # append 0.001 for all missing nodes
            self._all_features.append(np.ones((graphs.nodes_for_graph(graphs.name_to_index(graph)) - m.shape[0],
                                               m.shape[1])) * 0.001)
        # create one big matrix of everything - rows: nodes, columns: features
        self._all_features = np.concatenate(self._all_features)

        # all_ftr_graph_index - [ .... last_row_index_for_graph_i ... ]
        self._all_ftr_graph_index = np.cumsum([0] + graphs.nodes_count_list()).tolist()

        # count total nodes and total features
        self.num_nodes, self._num_features = self._all_features.shape
        super(AllBetaPairs, self).__init__(graphs, None)

    def _calc_beta(self, gid):
        beta_vec = []
        # get features matrix for interval
        g_index = self._graphs.name_to_index(gid)

        # cut the relevant part from the matrix off all features according to the interval size (and graph sizes)
        if g_index < self._interval:
            context_matrix = self._all_features[0: int(self._all_ftr_graph_index[self._interval]), :]
        else:
            context_matrix = self._all_features[int(self._all_ftr_graph_index[g_index - self._interval]):
                                                int(self._all_ftr_graph_index[g_index]), :]
        # get features matrix only for current graph
        g_matrix = self._graphs.features_matrix(gid)

        # for every one of the selected features: ftr_i, ftr_j
        # cf_window: coefficient of the linear regression on ftr_i/j in the in the window [curr - interval: current]
        # g_ftr_i: [ .... feature_i for node j .... ]
        # beta_vec = [ ....  mean < g_ftr_j - cf_window * g_ftr_i > .... ]
        # for i, j in self._ftr_pairs:
        for i in range(self._num_features):
            for j in range(i + 1, self._num_features):
                beta_vec.append(np.mean(g_matrix[:, j] -
                                linear_model.LinearRegression().fit(np.transpose(context_matrix[:, i].T),
                                                                    np.transpose(context_matrix[:, j].T)).coef_[0][0] *
                                g_matrix[:, i]))
        return np.asarray(beta_vec)
