from datetime import timedelta, datetime
import networkx as nx
import pandas as pd
from os import path
import pickle
import os

from beta_calculator import LinearContext
from feature_calculators import FeatureMeta
from features_picker import PearsonFeaturePicker
from loggers import PrintLogger
from norm_functions import log_norm
from timed_graphs import TimedGraphs
from vertices.attractor_basin import AttractorBasinCalculator
from vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from vertices.betweenness_centrality import BetweennessCentralityCalculator
from vertices.bfs_moments import BfsMomentsCalculator
from vertices.closeness_centrality import ClosenessCentralityCalculator
from vertices.communicability_betweenness_centrality import CommunicabilityBetweennessCentralityCalculator
from vertices.eccentricity import EccentricityCalculator
from vertices.fiedler_vector import FiedlerVectorCalculator
from vertices.flow import FlowCalculator
from vertices.general import GeneralCalculator
from vertices.hierarchy_energy import HierarchyEnergyCalculator
from vertices.k_core import KCoreCalculator
from vertices.load_centrality import LoadCentralityCalculator
from vertices.louvain import LouvainCalculator
from vertices.motifs import nth_nodes_motif

ANOMALY_DETECTION_FEATURES = {
    "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),  # directed only
    "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),
    "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
    "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),
    "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
    # "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator,
    #                                                       {"communicability"}),
    "eccentricity": FeatureMeta(EccentricityCalculator, {"ecc"}),
    "fiedler_vector": FeatureMeta(FiedlerVectorCalculator, {"fv"}),
    # "flow": FeatureMeta(FlowCalculator, {}),
    "general": FeatureMeta(GeneralCalculator, {"gen"}),
    # Isn't OK - also in previous version
    # "hierarchy_energy": FeatureMeta(HierarchyEnergyCalculator, {"hierarchy"}),
    "k_core": FeatureMeta(KCoreCalculator, {"kc"}),
    "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load_c"}),
    "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
    "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),
    # "multi_dimensional_scaling": FeatureMeta(MultiDimensionalScalingCalculator, {"mds"}),
    # "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),
    "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),
    # "first_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(1), {"fnh", "first_neighbor"}),
    # "second_neighbor_histogram": FeatureMeta(nth_neighbor_calculator(2), {"snh", "second_neighbor"}),
}

SOURCE = 'SourceID'
DEST = 'DestinationID'
DURATION = 'Duration'
TIME = 'StartTime'
COMMUNITY = 'Community'
TARGET = 'target'

ALL_BETA_PATH = "all_times_beta.pkl"


class RefaelDataLoader:
    def __init__(self ,data_name, data_path, params, days_split=1):
        self._target_path = os.path.join("data_by_time", data_name, "split_" + str(days_split))
        self._params = params
        self._logger = PrintLogger(self._params['logger_name'])
        self._params['files_path'] = self._target_path
        self._data_name = data_name
        self._data_path = data_path
        self._time_split=days_split
        self._partition_data()
        self._timed_graph = self._init_timed_graph()

        for i in range(10):
            self._forward_time()

        # self.calc_all_times()
        self._time_idx = 0

    def calc_all_times(self):
        if os.path.exists(ALL_BETA_PATH):
            self._all_times_data = pickle.load(open(ALL_BETA_PATH, "rb"))
            return
        self._all_times_data = []
        while self._forward_time():
            self._all_times_data.append([self._calc_curr_time()])
        pickle.dump(self._all_times_data, open(ALL_BETA_PATH, "wb"))

    def _init_timed_graph(self):
        return TimedGraphs(self._params['database'], files_path=self._params['files_path'], logger=self._logger,
                           features_meta=ANOMALY_DETECTION_FEATURES, directed=self._params['directed'],
                           date_format=self._params['date_format'], largest_cc=self._params['max_connected'])

    def _partition_data(self):
        if "data_by_time" not in os.listdir("."):
            os.mkdir("data_by_time")
        if self._data_name not in os.listdir("data_by_time"):
            os.mkdir(os.path.join("data_by_time", self._data_name))
        if "split_" + str(self._time_split) not in os.listdir(os.path.join("data_by_time", self._data_name)):
            os.mkdir(os.path.join("data_by_time", self._data_name, "split_" + str(self._time_split)))

        data_df = pd.read_csv(self._data_path)
        self._format_data(data_df)
        data_df = data_df.set_index(TIME).sort_index()

        one_day = timedelta(days=self._time_split)
        curr_day = data_df.first_valid_index()
        file_time = open(path.join(self._target_path, str(curr_day.date())), "wt")
        next_day = curr_day - timedelta(hours=curr_day.hour, minutes=curr_day.minute, seconds=curr_day.second) + one_day
        for curr_day, row in data_df.iterrows():
            if curr_day >= next_day:
                file_time.close()
                next_day = curr_day - timedelta(hours=curr_day.hour, minutes=curr_day.minute,
                                                seconds=curr_day.second) + one_day
                file_time = open(path.join(self._target_path, str(curr_day.date())), "wt")
            file_time.write(str(row[SOURCE]) + " " + str(row[DEST]) + " " + str(row[DURATION]) + " "
                            + str(row[COMMUNITY]) + " " + str(row[TARGET]) + "\n")


    @staticmethod
    def _format_data(graph_df):
        graph_df[TIME] = graph_df[TIME]/1000                                           # milliseconds to seconds
        graph_df[TIME] = graph_df[TIME].apply(lambda x: datetime.fromtimestamp(x))     # to datetime format

    def _forward_time(self):
        flag = self._timed_graph.forward_time()
        # normalize features ---------------------------------
        self._timed_graph.norm_features(log_norm)
        return flag

    def _calc_curr_time(self):
        pearson_picker = PearsonFeaturePicker(self._timed_graph, size=self._params['ftr_pairs'],
                                              logger=self._logger, identical_bar=self._params['identical_bar'])
        best_pairs = pearson_picker.best_pairs()
        beta = LinearContext(self._timed_graph, best_pairs, split=self._params['context_beta'])
        return beta.beta_matrix(), best_pairs, self._timed_graph.nodes_count_list(), \
               self._timed_graph.edges_count_list(), self._timed_graph.get_labels()

    def calc_curr_time(self):

        return self._all_times_data[self._time_idx - 1]

    def forward_time(self):
        if self._time_idx == len(self._all_times_data):
            return False
        self._time_idx += 1
        return True