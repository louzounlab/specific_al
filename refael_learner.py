from DataLoader.refael_data_loader import RefaelDataLoader
import os
from active_learning import ActiveLearning, DistType
from ml_communities import MLCommunities, LearningMethod


class RefaelLearner:
    def __init__(self):
        self._params = {
            'logger_name': "logger",
            # Data parameters
            'database': 'Refael',
            'data_file_name': 'Refael_07_18.csv',  # should be in ../data/
            'date_format': "%Y-%m-%d",  # Refael
            'directed': True,
            # features + beta vectors parameters
            'max_connected': False,
            'ftr_pairs': 200,
            'identical_bar': 0.99,
            'context_beta': 1,
            # ML- parameters
            'learn_method': LearningMethod.RF,
            # AL - parameters
            'eps': 0.01,
            'target_recall': 0.7,
            'dist_type': DistType.Euclidian
        }
        self._database = RefaelDataLoader(os.path.join("..", "data", self._params['data_file_name']), self._params)
        self._ml_learner = MLCommunities(method=self._params['learn_method'])
        self._al_learner = ActiveLearning(self._params)

    def run_ml(self):
        time = 0
        while self._database.forward_time():
            print("-----------------------------------    TIME " + str(time) + "    ----------------------------------")
            time += 1
            beta_matrix, best_pairs, nodes_list, edges_list, labels = self._database.calc_curr_time()
            self._ml_learner.forward_time_data(beta_matrix, best_pairs, nodes_list, edges_list, labels)
            self._ml_learner.run()

    def run_al(self):
        time = 0
        while self._database.forward_time():
            print("-----------------------------------    TIME " + str(time) + "    ----------------------------------")
            time += 1
            beta_matrix, best_pairs, nodes_list, edges_list, labels = self._database.calc_curr_time()
            self._al_learner.forward_time_data(beta_matrix, best_pairs, nodes_list, edges_list, labels)
            self._al_learner.run()


if __name__ == "__main__":
    r = RefaelLearner()
    r.run_al()
