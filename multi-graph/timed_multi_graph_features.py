

"""
    all the graphs are stored under one graph
    - each edge has a set of attributes - {graph_1: weight_1, ...... graph_k: weight_k}
    - if the database files name are with dates, use date_format='format' for example date_format='%d-%b-%Y'
"""
import datetime
from graph_features import GraphFeatures
from feature_meta import NODE_FEATURES
from loggers import PrintLogger
import networkx as nx
import os
import numpy as np
from os import path


SOURCE = 'SourceID'
DEST = 'DestinationID'
DURATION = 'Duration'
TIME = 'StartTime'
COMMUNITY = 'Community'
TARGET = 'target'


class TimedMultiGraphFeatures:
    def __init__(self, database_name, logger: PrintLogger, features_meta=None,
                 directed=False, files_path=None, date_format=None, pkl_dir=None):
        self._features_meta = NODE_FEATURES if features_meta is None else features_meta
        self._pkl_dir = pkl_dir if pkl_dir else path.join("data", database_name)
        # define func for sorting graphs
        if date_format is None:
            self._key_func = None
        else:
            self._key_func = lambda x: datetime.datetime.strptime(x, date_format)

        self._labels_dict = {}
        self._list_id = []  # list of graph ID's - gives a mapping index to ID
        self._dict_id = {}  # dictionary of graph ID's - gives a mapping ID to index
        self._directed = directed
        self._gnx_multi = nx.DiGraph() if directed else nx.Graph()
        self._path = os.path.join('data', database_name)
        self._logger = logger
        self._files_path = files_path  # location of graphs as files
        self._logger.debug("finish initialization")
        self._features_matrix_dict = {}
        self._features_calculated = False
        self._initiation()

    def get_feature_meta(self):
        return self._features_meta

    @staticmethod
    def _strip_txt(file_name):
        return file_name.split(".")[0]

    def _initiation(self):
        if not self._files_path or not os.path.exists(self._files_path):
            self._logger.error("\n\n\t\t*** THE PATH TO THE DATABASE FILES IS NOT VALID !!! ***\n")
            exit(1)

        self._times = [graph for graph in sorted(os.listdir(self._files_path), key=self._key_func)]
        self._current_time = 0
        self._total_time = len(self._times)
        self._graph_index = 0
        self._changed_communities = []

    def forward_time(self):
        self._changed_communities = []
        # check time is legal - stops when reaches END
        if self._current_time == self._total_time:
            return False

        time_name = self._strip_txt(self._times[self._current_time])
        self._current_time += 1
        time_file = open(os.path.join(self._files_path, time_name))

        for row in time_file:
            [node_u, node_v, duration, community, target] = row.split()

            # add community to updated
            if community not in self._changed_communities:
                self._changed_communities.append(community)

            # add community as subgraph
            if community not in self._list_id:
                self._list_id.append(community)
                self._dict_id[community] = self._graph_index
                self._graph_index += 1

            self._logger.debug("adding edge:\t(" + str(node_u) + "," + str(node_v) + ")\ttime=" + str(time_name) + "\t["
                               + str(community) + "," + str(duration) + "," + str(target) + "]")
            # add the edge to the graph if it doesn't exist
            self._gnx_multi.add_edge(node_u, node_v)

            # build Graph - for each edge -> { ... COMMUNITY: {TARGET, COMMUNITY ...}
            node_info = {DURATION: duration, TARGET: target}
            self._gnx_multi.edges[node_u, node_v][community] = node_info
            self._labels_dict[community] = int(target)
        return True

    def get_labels(self):
        labels_list = []
        for com in self._list_id:
            labels_list.append(self._labels_dict[com])
        return labels_list

    def subgraph_by_name(self, graph_name: str):
        if not self.is_graph(graph_name):
            self._logger.error("no graph named:\t" + graph_name)
            return
        subgraph_edges = nx.DiGraph() if self._directed else nx.Graph()
        for edge in list(self._gnx_multi.edges(data=True)):
            if graph_name in edge[2]:
                # edge is saved in the following method (from, to, {graph_name_i: weight_in_graph_i})
                subgraph_edges.add_edge(edge[0], edge[1])
        return subgraph_edges

    def subgraph_by_index(self, index: int):
        if index < 0 or index > len(self._list_id):
            self._logger.error("index is out of scope - index < number_of_graphs/ index > 0")
        return self.subgraph_by_name(self._list_id[index])

    @staticmethod
    def _one_from_list_in_dict(names_list, names_dict):
        for name in names_list:
            if name in names_dict:
                return True
        return False

    def combined_graph_by_names(self, names_list=None, combine_all=False):
        # case combine_all - the returned graph will combination of all graphs regardless to the names list
        if combine_all:
            names_list = self._list_id
        else:
            for graph_name in names_list:
                if not self.is_graph(graph_name):
                    self._logger.error("no graph named:\t" + graph_name)
                    return
        subgraph_edges = nx.Graph()
        for edge in list(self._gnx_multi.edges(data=True)):
            if self._one_from_list_in_dict(names_list, edge[2]):
                # edge is saved in the following method (from, to, {graph_name_i: weight_in_graph_i})
                subgraph_edges.add_edge(edge[0], edge[1])
        return subgraph_edges

    def combined_graph_by_indexes(self, index_list=None, combine_all=False):
        i = 0
        for index in index_list:
            if index < 0 or index > len(self._list_id):
                self._logger.error("index is out of scope - index < number_of_graphs/ index > 0")
                return
            else:
                index_list[i] = self._list_id[index]
                i += 1
        return self.combined_graph_by_names(index_list, combine_all)

    def is_graph(self, graph_name):
        return True if graph_name in self._dict_id else False

    def index_to_name(self, index):
        if index < 0 or index > len(self._list_id):
            self._logger.error("index is out of scope - index < number_of_graphs/ index > 0")
            return None
        return self._list_id[index]

    def name_to_index(self, graph_name):
        if not self.is_graph(graph_name):
            self._logger.error("no graph named:\t" + graph_name)
            return None
        return self._dict_id[graph_name]

    def build_features(self, largest_cc=False, should_zscore=True):
        for community in self._changed_communities:
            self._logger.debug("calculating features for " + community)
            gnx_path = os.path.join(self._pkl_dir, community)
            if community not in os.listdir(self._pkl_dir):
                os.mkdir(gnx_path)
            gnx = self.subgraph_by_name(community)
            gnx_ftr = GraphFeatures(gnx, self._features_meta, dir_path=gnx_path, logger=self._logger,
                                    is_max_connected=largest_cc)
            gnx_ftr.build(should_dump=False, force_build=True)  # build ALL_FEATURES
            self._features_matrix_dict[community] = gnx_ftr.to_matrix(dtype=np.float32, mtype=np.matrix,
                                                                     should_zscore=should_zscore)

    def features_matrix_by_indexes(self, graph_start=0, graph_end=0, for_all=False):
        if for_all:
            graph_start = 0
            graph_end = len(self._list_id) - 1

        np_matrix = self._features_matrix_dict[self.index_to_name(graph_start)]
        graph_index = graph_start + 1
        while graph_index < graph_end:
            np_matrix = np.concatenate((np_matrix, self._features_matrix_dict[self.index_to_name(graph_index)]))
            graph_index += 1
        return np_matrix

    def features_matrix_by_names(self, graph_start=0, graph_end=0, for_all=False):
        if for_all:
            return self.features_matrix_by_indexes(0, len(self._list_id), for_all=for_all)
        if not self.is_graph(graph_start):
            self._logger.error("no graph named:\t" + str(graph_start))
        if not self.is_graph(graph_end):
            self._logger.error("no graph named:\t" + str(graph_end))
        return self.features_matrix_by_indexes(self.name_to_index(graph_start), self.name_to_index(graph_end), for_all=for_all)

    def feature_matrix(self, name_index):
        return self._features_matrix_dict[self.index_to_name(name_index)] if type(name_index) is int else \
            self._features_matrix_dict[name_index]

    def nodes_for_graph(self, name_index):
        return self.subgraph_by_index(name_index).number_of_nodes() if type(name_index) is int else \
            self.subgraph_by_name(name_index).number_of_nodes()

    def nodes_count_list(self):
        nodes_count = []
        for community in self._list_id:
            nodes_count.append(self.nodes_for_graph(community))
        return nodes_count

    def edges_for_graph(self, name_index):
        return self.subgraph_by_index(name_index).number_of_edges() if type(name_index) is int else \
            self.subgraph_by_name(self.name_to_index(name_index)).number_of_edges()

    def edges_count_list(self):
        nodes_count = []
        for community in self._list_id:
            nodes_count.append(self.edges_for_graph(community))
        return nodes_count

    def subgraphs(self, start_id=None, end_id=None):
        for gid in self._list_id[start_id: end_id]:
            yield self.subgraph_by_name(gid)

    def graph_names(self, start_id=None, end_id=None):
        for gid in self._list_id[start_id: end_id]:
            yield gid

    def number_of_graphs(self):
        return len(self._list_id)

    def norm_features(self, norm_function):
        for M in self._features_matrix_dict:
            self._features_matrix_dict[M] = norm_function(self._features_matrix_dict[M])

