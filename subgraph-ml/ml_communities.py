import os
from enum import Enum

import feature_meta
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedShuffleSplit
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

REBUILD_FEATURES = False
RE_PICK_FTR = False

CHOSEN_FEATURES = feature_meta.NODE_FEATURES
# CHOSEN_FEATURES = {"multi_dimensional_scaling": FeatureMeta(MultiDimensionalScaling, {"mds"})}
PATH = os.path.join("..", "data_by_community")


class LearningMethod(Enum):
    RF = "random_forest"
    SVM = "support_vector_machine"


class MLCommunities:
    def __init__(self, method=LearningMethod.RF):
        self.labels = None
        self._beta_pairs = None
        self._beta_matrix = None
        self._nodes = None
        self._edges = None
        self._best_beta_df = None
        self._method = method

    def forward_time_data(self, beta_matrix, best_pairs, nodes, edges, labels):
        self.labels = labels
        self._beta_pairs = best_pairs
        self._beta_matrix = beta_matrix
        self._nodes = nodes
        self._edges = edges
        # self._best_beta_df = self._best_pairs_df()
        self._best_beta_df = self._beta_matrix_to_df(self._beta_pairs)

    def run(self):
        if self._method == LearningMethod.RF:
            res_df = self._learn_RF(self._pca_df(self._best_beta_df, graph_data=True, min_nodes=10))
        if self._method == LearningMethod.SVM:
            self._learn_SVM(self._pca_df(self._best_beta_df, graph_data=True, min_nodes=5))

    def _beta_matrix_to_df(self, header):
        # create header
        header.append("labels")
        return pd.DataFrame(data=np.hstack((self._beta_matrix, np.matrix(self.labels).T)), columns=header)

    def _pca_df(self, beta_df, n_components=20, graph_data=False, min_nodes=None):
        pca = PCA(n_components=n_components)

        if min_nodes:
            beta_df_temp = beta_df.copy()
            beta_df_temp['nodes'] = self._nodes
            beta_df_temp['edges'] = self._edges
            beta_df_temp['labels'] = self.labels
            beta_df_temp = beta_df_temp[beta_df_temp.nodes >= min_nodes]
            self.labels = beta_df_temp['labels'].tolist()
            self._nodes = beta_df_temp['nodes'].tolist()
            self._edges = beta_df_temp['edges'].tolist()
            beta_df_temp = beta_df_temp.drop(['nodes', 'labels'], axis=1)
            beta_df = beta_df_temp

        if graph_data:
            # add edge and node number
            return np.hstack([pca.fit_transform(beta_df), np.matrix(self._nodes).T, np.matrix(self._edges).T])

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
                cv = StratifiedShuffleSplit(n_splits=1, test_size=1 - float(train_p) / 100)
                clf_rf = RandomForestClassifier(n_estimators=200, max_features="log2", criterion="gini", max_depth=max_depth)
                # print(clf_svm)
                # clf_RF = RandomForestClassifier()
                scores_svm = cross_val_score(clf_rf, principalComponents, self.labels, cv=cv, scoring='roc_auc')
                # scores_rf = cross_val_score(clf_RF, self._conf.beta_matrix, self._conf.labels, cv=cv)
                df.loc[len(df)] = [max_depth, train_p, np.mean(scores_svm)]
                print([max_depth, train_p, np.mean(scores_svm)])
        return df

    def plot_learning_df(self, df):
        new_df = pd.DataFrame(df[df['rf-max_depth'] == 9], )
        new_df.reset_index()
        new_df.plot(x='train_p', y='mean_auc')
        plt.savefig("auc.jpg")
        plt.show()
