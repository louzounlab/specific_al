from enum import Enum
from beta_calculator import LinearContext
from feature_calculators import FeatureMeta
from features_picker import PearsonFeaturePicker
from graphs_al import Graphs
from loggers import PrintLogger
from norm_functions import log_norm
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
from vertices.multi_dimensional_scaling import MultiDimensionalScalingCalculator
from vertices.page_rank import PageRankCalculator

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

REBUILD_FEATURES = False
RE_PICK_FTR = False



class DistType(Enum):
    Euclidian = "euclidean"
    OneClass = "one_class"

class ParameterConf:
    def __init__(self, path, dist_type=DistType.Euclidian, eps=0.01, recall=0.7):

        self._params = {
            'database': 'Refael',
            'files_path': path,
            'date_format': None,  # Twitter
            'directed': True,
            'max_connected': False,
            'logger_name': "logger",
            'ftr_pairs': 300,
            'identical_bar': 0.9,
            'context_beta': 1,
        }

        # self._labels = []
        # self._beta_matrix = None
        self.eps = eps
        self.recall = recall
        self.dit_type = dist_type

        self._logger = PrintLogger(self._params['logger_name'])
        self._graphs = Graphs(self._params['database'], files_path=self._params['files_path'], logger=self._logger,
                              features_meta=ANOMALY_DETECTION_FEATURES, directed=self._params['directed'],
                              date_format=self._params['date_format'], largest_cc=self._params['max_connected'])
        self._graphs.build(force_rebuild_ftr=REBUILD_FEATURES, pick_ftr=RE_PICK_FTR, should_zscore=False)
        self.labels = self._graphs.get_labels()

        # normalize features ---------------------------------
        self._graphs.norm_features(log_norm)

        pearson_picker = PearsonFeaturePicker(self._graphs, size=self._params['ftr_pairs'],
                                              logger=self._logger, identical_bar=self._params['identical_bar'])
        best_pairs = pearson_picker.best_pairs()
        beta = LinearContext(self._graphs, best_pairs, split=self._params['context_beta'])
        self.beta_matrix = beta.beta_matrix()

