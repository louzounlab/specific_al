import os

from ParametersConf import ANOMALY_DETECTION_FEATURES
from beta_calculator import LinearContext
from beta_correlation import AllBetaPairs
from features_picker import PearsonFeaturePicker
from graphs_al import Graphs
from loggers import PrintLogger
import feature_meta
from norm_functions import log_norm
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

REBUILD_FEATURES = False
RE_PICK_FTR = False

CHOSEN_FEATURES = feature_meta.NODE_FEATURES
# CHOSEN_FEATURES = {"multi_dimensional_scaling": FeatureMeta(MultiDimensionalScaling, {"mds"})}
PATH = os.path.join("..", "data_by_community")  # IMPORTANT!! - PATH TO THE DATA
BETA_PKL_P = "pearson_best_beta.pkl"


class BasicLearner:
    def __init__(self, path, eps=0.01, recall=0.7):

        self._params = {
            'database': 'Refael',
            'files_path': path,
            'date_format': None,  # Twitter
            'directed': True,
            'max_connected': False,
            'logger_name': "logger",
            'ftr_pairs': 300,
            'identical_bar': 0.95,
            'context_beta': 1,
        }

        self._logger = PrintLogger(self._params['logger_name'])
        self._graphs = Graphs(self._params['database'], files_path=self._params['files_path'], logger=self._logger,
                              features_meta=ANOMALY_DETECTION_FEATURES, directed=self._params['directed'],
                              date_format=self._params['date_format'], largest_cc=self._params['max_connected'])
        self._graphs.build(force_rebuild_ftr=REBUILD_FEATURES, pick_ftr=RE_PICK_FTR, should_zscore=False)

        # normalize features ---------------------------------
        self._graphs.norm_features(log_norm)

        # labels
        self.labels = self._graphs.get_labels()

        pearson_picker = PearsonFeaturePicker(self._graphs, size=self._params['ftr_pairs'],
                                              logger=self._logger, identical_bar=self._params['identical_bar'])
        best_pairs = pearson_picker.best_pairs()
        self._pairs_header = best_pairs

        if os.path.exists(BETA_PKL_P):
            self._beta_matrix = pickle.load(open(BETA_PKL_P, "rb"))
        else:
            beta = LinearContext(self._graphs, best_pairs, split=self._params['context_beta'])
            self._beta_matrix = beta.beta_matrix()
            pickle.dump(self._beta_matrix, open(BETA_PKL_P, "wb"))

        self._beta_df = self._beta_matrix_to_df(header=self._pairs_header)
        # self._best_beta_df = self._best_pairs_df()
        self._best_beta_df = self._beta_df
        res_df = self._learn_RF(self._pca_df(self._best_beta_df, graph_data=True, min_nodes=10))
        self.plot_learning_df(res_df)
        # self._learn_SVM(self._pca_df(self._best_beta_df, graph_data=True, min_nodes=5))

    def _beta_matrix_to_df(self, header=None):
        # create header
        if not header:
            header = []
            for i in range(223):
                for j in range(i + 1, 223):
                    header.append("(" + str(i) + ", " + str(j) + ")")
        header.append("labels")
        return pd.DataFrame(data=np.hstack((self._beta_matrix, np.matrix(self.labels).T)), columns=header)

    def _best_pairs_df(self):
        # limited df, on the clumns there are only pairs with |correlation coeff| >= 0.2
        best_beta_df = pd.DataFrame()
        # best_pairs_info = []

        for name in self._beta_df:
            # don't compare labels with itself
            if not name == "labels":
                # correlation of each column with labels
                cor = self._beta_df[name].corr(self._beta_df["labels"])
                # take all pairs with |correlation coeff| >= 0.2
                if np.abs(cor) >= 0.2:
                    best_beta_df[name] = self._beta_df[name]
        return best_beta_df

    def _pca_df(self, beta_df, n_components=20, graph_data=False, min_nodes=None):
        pca = PCA(n_components=n_components)
        nodes_list = self._graphs.nodes_count_list()
        edge_list = []
        for graph in self._graphs.subgraphs():
            edge_list.append(graph.number_of_edges())

        if min_nodes:
            beta_df_temp = beta_df.copy()
            beta_df_temp['nodes'] = nodes_list
            beta_df_temp['edges'] = edge_list
            beta_df_temp['labels'] = self.labels
            beta_df_temp = beta_df_temp[beta_df_temp.nodes >= min_nodes]
            self.labels = beta_df_temp['labels'].tolist()
            nodes_list = beta_df_temp['nodes'].tolist()
            edge_list = beta_df_temp['edges'].tolist()
            beta_df_temp = beta_df_temp.drop(['nodes', 'labels'], axis=1)
            beta_df = beta_df_temp

        if graph_data:
            # add edge and node number
            return np.hstack([pca.fit_transform(beta_df), np.matrix(nodes_list).T, np.matrix(edge_list).T])

        return pca.fit_transform(beta_df)

    def _learn_SVM(self, principalComponents):
        df = pd.DataFrame(columns=['C', 'train_p', 'mean_auc'])
        # penalty for svm
        for C in np.logspace(-2, 2, 5):
            # train percentage
            for train_p in range(5, 90, 10):
                cv = ShuffleSplit(n_splits=1, test_size=1 - float(train_p) / 100)
                clf_svm = SVC(C=C, kernel='linear', probability=False, shrinking=False,
                              class_weight='balanced')
                # print(clf_svm)
                # clf_RF = RandomForestClassifier()
                scores_svm = cross_val_score(clf_svm, principalComponents, self.labels, cv=cv, scoring='roc_auc')
                # scores_rf = cross_val_score(clf_RF, self._conf.beta_matrix, self._conf.labels, cv=cv)
                df.loc[len(df)] = [C, train_p, np.mean(scores_svm)]
                print([C, train_p, np.mean(scores_svm)])
        return df

    def _learn_RF(self, principalComponents):
        df = pd.DataFrame(columns=['rf-max_depth', 'train_p', 'mean_auc'])
        # train percentage
        for train_p in range(70, 90, 5):
            for max_depth in range(10, 25):
                cv = ShuffleSplit(n_splits=4, test_size=1 - float(train_p) / 100)
                clf_rf = RandomForestClassifier(n_estimators=200, max_features="log2", criterion="gini", max_depth=max_depth)
                # print(clf_svm)
                # clf_RF = RandomForestClassifier()
                scores_svm = cross_val_score(clf_rf, principalComponents, self.labels, cv=cv, scoring='roc_auc')
                # scores_rf = cross_val_score(clf_RF, self._conf.beta_matrix, self._conf.labels, cv=cv)
                df.loc[len(df)] = [max_depth, train_p, np.mean(scores_svm)]
                print([max_depth, train_p, np.mean(scores_svm)])
        df.to_csv("with_10_plus.csv")
        return df

    def plot_learning_df(self, df):
        new_df = pd.DataFrame(df[df['rf-max_depth'] == 9], )
        new_df.reset_index()
        new_df.plot(x='train_p', y='mean_auc')
        plt.savefig("auc.jpg")
        plt.show()


if __name__ == "__main__":
    d = BasicLearner(PATH)
    b = 0
