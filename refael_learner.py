import pickle
from DataLoader.refael_data_loader import RefaelDataLoader
import os
from ParametersConf import DistType
from active_learning import ActiveLearning
from ml_communities import MLCommunities, LearningMethod


class RefaelLearner:
    def __init__(self):
        self._params = {
            'logger_name': "logger",
            # Data parameters
            'days_split': 1,
            'start_interval': 10,
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
            'batch_size': 5,
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
        # plot results y_axis
        steps_to_recall = []
        # plot results x_axis
        times = []
        time = 0
        while self._database.forward_time():
            print("-----------------------------------    TIME " + str(time) + "    ----------------------------------")
            beta_matrix, best_pairs, nodes_list, edges_list, labels = self._database.calc_curr_time()
            self._al_learner.forward_time_data(beta_matrix, best_pairs, nodes_list, edges_list, labels)
            times.append(time)
            steps_to_recall.append(self._al_learner.run())
            time += 1

        # save results to pkl
        pickle.dump([times, steps_to_recall], open("al_results_split_" + self._params['start_interval'] +
                                                   "_start_" + self._params['start_interval'] +
                                                   "_batch_" + self._params['batch_size'] + ".pkl", "wb"))
        return [times, steps_to_recall]

    def run_al_simulation(self):
        results = {}
        for split_interval in range(1, 4):
            for batch_size in range(1, 21):
                self._params['days_split'] = split_interval
                self._params['batch_size'] = batch_size
                self._params['start_interval'] = 12 / split_interval
                results[(split_interval, batch_size)] = self.run_al()[1]

        pickle.dump(results, open("simulation_results.pkl", "wb"))


if __name__ == "__main__":
    r = RefaelLearner()
    r.run_al_simulation()
    # r.run_al()
