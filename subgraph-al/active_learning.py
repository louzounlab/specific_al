from enum import Enum

from explore_exploit import ExploreExploit
import pandas as pd


class DistType(Enum):
    Euclidian = "euclidean"
    OneClass = "one_class"


class ActiveLearning:
    def __init__(self, params):
        self._eps = params['eps']
        self._target_recall = params['target_recall']
        self._dist_type = params['dist_type']
        self.labels = None
        self._beta_pairs = None
        self._beta_matrix = None
        self._nodes = None
        self._edges = None
        self._best_beta_df = None

    def forward_time_data(self, beta_matrix, best_pairs, nodes, edges, labels):
        self.labels = labels
        self._beta_pairs = best_pairs
        self._beta_matrix = beta_matrix
        self._nodes = nodes
        self._edges = edges

    def run(self):
        # one_class - most anomal node
        # euclidean - the node that
        for eps in [0, 0.01, 0.05]:
            mean_steps = 0
            time_tag_dict = {}
            for i in range(1, 11):
                # number of average steps for recall 0.7 black from all blacks
                exploration = ExploreExploit(self.labels, self._beta_matrix, self._target_recall, self._eps)
                num_steps, tags = exploration.run(self._dist_type)
                print(" an recall of 70% was achieved in " + str(num_steps) + " steps")
                mean_steps += num_steps
                time_tag_dict[i] = tags
            time_tag_df = pd.DataFrame.from_dict(time_tag_dict, orient="index").transpose()
            time_tag_df.to_csv(self._dist_type.name + "_output.csv")
            print("the mean num of steps is: " + str(mean_steps / 10))
