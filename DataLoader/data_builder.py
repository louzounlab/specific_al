from datetime import timedelta, datetime
from statsmodels import duration
import networkx as nx
import time
import pandas as pd
from os import path
import pickle
import os
SOURCE = 'SourceID'
DEST = 'DestinationID'
DURATION = 'Duration'
TIME = 'StartTime'
COMMUNITY = 'Community'
TARGET = 'target'


class Data:
    def __init__(self, path):
        self._communities = []
        self._gnx = nx.DiGraph()
        # load csv and sort by time
        self._graph_df = pd.read_csv(path)
        self._graph_df.sort_values([COMMUNITY], ascending=True)

        del self._graph_df[DURATION]
        del self._graph_df[TIME]

        # self._format_data(self._graph_df)     # format time
        # self._build_graph(graph_df)         # build graph

    def split_to_files(self):
        # set the index to be this and don't drop
        self._graph_df.set_index(keys=[COMMUNITY], drop=False, inplace=True)
        # get a list of names
        community_list = self._graph_df[COMMUNITY].unique().tolist()
        # now we can perform a lookup on a 'view' of the dataframe

        for c in community_list:
            df = self._graph_df.loc[self._graph_df[COMMUNITY] == c]
            del df[COMMUNITY]
            df.to_csv(os.path.join("data_by_community", str(c) + ".txt"), index=False, header=False, sep=" ")

    def _build_graph(self, graph_df):
        # build Graph - for each edge -> { ... TIME: {DURATION, COMMUNITY, TARGET} ...}
        for i in range(len(graph_df)):
            self._gnx.add_edge(graph_df.loc[i, :][SOURCE], graph_df.loc[i, :][DEST])  # add edge to graph
            current_edge_dict = {DURATION: graph_df.loc[i, :][DURATION],              # build edge attribute dictionary
                                 COMMUNITY: graph_df.loc[i, :][COMMUNITY],
                                 TARGET: graph_df.loc[i, :][TARGET]}
            self._gnx.edges[graph_df.loc[i, :][SOURCE], graph_df.loc[i, :][DEST]][graph_df.loc[i, :][TIME]] = current_edge_dict
            self._communities.append(graph_df.loc[i, :][COMMUNITY] if graph_df.loc[i, :][COMMUNITY] not in self._communities
                                     else None)

    def subgraph(self, community):
        new_graph = nx.DiGraph()
        for edge in self._gnx.edges(data=True):
            is_edge_relevant = False
            # seek for community in edge
            for start_time in edge[2]:
                if community in start_time[COMMUNITY]:
                    is_edge_relevant = True
            if is_edge_relevant:
                new_graph.add_edge(edge)
        return new_graph

    def dump(self):
        pickle.dump(self._gnx, open("gnx_al.pkl", "wb"))

    def load(self):
        if os.path.exists("gnx_al.pkl"):
            self._gnx = pickle.load(open("gnx_al.pkl", "rb"))
            return True
        return False

    @staticmethod
    def _format_data(graph_df):
        graph_df[TIME] = graph_df[TIME]/1000                                           # milliseconds to seconds
        graph_df[TIME] = graph_df[TIME].apply(lambda x: datetime.fromtimestamp(x))     # to datetime format


if __name__ == "__main__":

    d = Data(path.join("data", "networks.csv"))
    d.split_to_files()
    b = 0

