import os
import pandas as pd
import networkx as nx
import numpy as np
from ParametersConf import ParameterConf
from graph_features import GraphFeatures
from loggers import PrintLogger
# from features_infra.feature_calculators import FeatureMeta
from explore_exploit import ExploreExploit
# from features_algorithms.vertices.multi_dimensional_scaling import MultiDimensionalScaling
import feature_meta
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


CHOSEN_FEATURES = feature_meta.NODE_FEATURES
# CHOSEN_FEATURES = {"multi_dimensional_scaling": FeatureMeta(MultiDimensionalScaling, {"mds"})}
PATH = os.path.join("..", "data_by_community")


class LoadData:
    def __init__(self):
        self._conf = ParameterConf(PATH)

    def run_eps_greedy(self):
        # one_class - most anomal node
        # euclidean - the node that
            for eps in [0, 0.01, 0.05]:
                mean_steps = 0
                time_tag_dict = {}
                for i in range(1, 11):
                    # number of average steps for recall 0.7 black from all blacks
                    exploration = ExploreExploit(self._conf.labels, self._conf.beta_matrix, self._conf.recall,
                                                 eps)
                    num_steps, tags = exploration.run(self._conf.dit_type.value)
                    print(" an recall of 70% was achieved in " + str(num_steps) + " steps")
                    mean_steps += num_steps
                    time_tag_dict[i] = tags
                time_tag_df = pd.DataFrame.from_dict(time_tag_dict, orient="index").transpose()
                time_tag_df.to_csv(self._conf.dit_type.value + "_output.csv")
                print("the mean num of steps is: " + str(mean_steps/10))

    def run_ML_algorithms(self):
        df = pd.DataFrame(columns=['C', 'kernel', 'probability', 'shrinking', 'class_weight',
                                   'train_p', 'mean_acc', 'std_acc'])
        for C in np.logspace(-3,3,6):
            for kernel in ['linear']:
                for probability in [False]:
                    for shrinking in [False]:
                        for class_weight in ['balanced']:
                            for train_p in range(5, 90, 10):
                                cv = ShuffleSplit(n_splits=3, test_size=1-train_p/100)
                                clf_svm = SVC(C=C, kernel=kernel, probability=probability, shrinking=shrinking,
                                              class_weight=class_weight)
                                # print(clf_svm)
                                # clf_RF = RandomForestClassifier()
                                scores_svm = cross_val_score(clf_svm, self._conf.beta_matrix, self._conf.labels, cv=cv)
                                # scores_rf = cross_val_score(clf_RF, self._conf.beta_matrix, self._conf.labels, cv=cv)
                                df.loc[len(df)] = [C, kernel, probability, shrinking, class_weight, train_p,
                                           np.mean(scores_svm), np.std(scores_svm)]
                                print([C, kernel, probability, shrinking, class_weight, train_p,
                                           np.mean(scores_svm), np.std(scores_svm)])
                                # print("svm scores: " + str(np.mean(scores_svm)))
                                # print("RF scores: " + str(np.mean(scores_rf)))
        print(df)

    def run_RF(self):
        df = pd.DataFrame(columns=['n_estimators', 'criterion', 'max_features', 'max_depth', 'min_samples_split',
                                   'train_p', 'mean_acc', 'std_acc'])
        for n_estimators in np.linspace(50, 200, 6):
            for criterion in ['gini', 'entropy']:
                for max_features in ['auto', 'log2', None]:
                    for max_depth in np.linspace(3, 50, 3):
                        for min_samples_split in np.linspace(2, 100, 5):
                            for train_p in range(5, 90, 10):
                                cv = ShuffleSplit(n_splits=3, test_size=1 - train_p / 100)
                                # clf_svm = SVC(C=C, kernel=kernel, probability=probability, shrinking=shrinking,
                                #               class_weight=class_weight)
                                # print(clf_svm)
                                clf_RF = RandomForestClassifier(oob_score=True, n_estimators=int(n_estimators),
                                                                criterion=criterion, max_features=max_features,
                                                                max_depth=max_depth,
                                                                min_samples_split=int(min_samples_split))
                                # scores_svm = cross_val_score(clf_svm, self._conf.beta_matrix, self._conf.labels, cv=cv)
                                scores_rf = cross_val_score(clf_RF, self._conf.beta_matrix, self._conf.labels, cv=cv)
                                df.loc[len(df)] = [n_estimators, criterion, max_features, max_depth, min_samples_split,
                                                   train_p, np.mean(scores_rf), np.std(scores_rf)]
                                print([n_estimators, criterion, max_features, max_depth, min_samples_split,
                                       train_p, np.mean(scores_rf), np.std(scores_rf)])
                                # print("svm scores: " + str(np.mean(scores_svm)))
                                # print("RF scores: " + str(np.mean(scores_rf)))
        print(df)


if __name__ == '__main__':
    dl = LoadData()
    # dl.run_eps_greedy()
    dl.run_ML_algorithms()
    print("bla")
