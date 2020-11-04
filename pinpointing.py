import datetime as dt
import logging
import os

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kando import kando_client
from scipy.stats import zscore
from tslearn.metrics import dtw_subsequence_path

bank = []

load_dotenv()

logging.basicConfig()

logger = logging.getLogger('pinpointing')
logger.setLevel(logging.DEBUG)

client = kando_client.client(os.getenv("KANDO_URL"), os.getenv("KEY"),
                             os.getenv("SECRET"))

scoring_weights = {
    'EC': 1,
    'PH': 1,
    'PI': 0,
    'TSS': 0.1,
    'COD': 0.1,
    'ORP': 0,
    'Battery': 0,
    'Signal': 0,
    'TEMPERATURE': 0.3
}


def _parser(node, graph):
    if len(node['children']) == 0:
        graph.add_node(node['point_id'], name=node['point']['name'])
        return
    for child in node['children']:
        graph.add_edges_from([(node['point_id'], child['point_id'])],
                             distance=child['parent_distance'])
        _parser(child, graph)


def get_graph(point):
    graph_data = client.network_graph(point)
    graph = nx.DiGraph()
    _parser(graph_data, graph)
    return graph


def _get_data(point, start, end):
    # convert time to epoch, fetch from API, turn into dataframe, return
    return_data = client.get_data(point_id=point,
                                  start=start.timestamp(),
                                  end=end.timestamp())
    samples = return_data['samplings']
    if len(samples) == 0:
        logger.info(f'No data at all for node {point}, continuing the search')
        return None
    df = pd.DataFrame(samples).T
    df.index = pd.to_datetime(df.index, unit='s')
    df = (df.drop(columns=['DateTime']).astype(float).resample(
        '1min').interpolate().resample('5min').mean())
    return df


def _get_dtw_distance(query, ts):
    return dtw_subsequence_path(query, ts)[1]


def _normalise(arr):
    return zscore(arr)


class Chain:
    # initiated with a list of one or multiple nodes
    def __init__(self, nodes):
        self.nodes = nodes

    def add(self, node):
        self.nodes.append(node)

    def get_score(self, scores):
        """
        Returns weighted average of non-NaN/non-None values from sensors, per weights (above)
        Final score is averaged by length of path (TODO maybe should be complement of length?)
        :param scores: dict of scores for each sensor of each node
        :return: weighted score of this node
        """
        res = 0
        node_score = [
            scores[node] for node in self.nodes[1:] if scores[node] is not None
        ]  # skip root
        for score in node_score:
            clean_score = {k: v for k, v in score.items() if not np.isnan(v)}
            res += sum(
                [v * scoring_weights[k]
                 for k, v in clean_score.items()]) / len(clean_score)
        return res / len(node_score)

    def remove_trailing_nones(self, scores):
        # if a chain ends in several Nones, remove them
        # the rationale is that we include None in the chain in case we have a match higher up
        # but we don't want to include Nones in the chain if we don't have data after them
        for idx, node in enumerate(self.nodes[::-1]):
            if scores[node] is None:
                continue
            return self.nodes[:len(self.nodes) - idx]


class Pinpointer:
    def __init__(self, root, graph, query, threshold):
        self.root = root
        self.graph = graph
        self.query = query
        self.threshold = threshold
        self.suspects = []
        self.scores = {}
        self.normalised_query = {}
        self.query_parameters = set(self.query)

        for param in self.query:
            if self.query[param].dtype == 'float64':
                if not all(self.query[param].isna()):
                    self.normalised_query[param] = _normalise(
                        self.query[param])

    def _get_time_shifted_data(self, node, path):
        """
        Given a chain and a node, and (start,end) times, find the equivalent time-shifted (start,end) for the node
        Note that we "pad" the time-shifting by 50% each way
        So for example if event takes place at 08:00-08:30 at event node, and candidate node is 3 hours behind,
        we will return data from start - (distance * 1.5), ie 03:30 until end - (distance * 0.5) ie 07:00
        Within this 3.5 hour space we will search for a pattern matching the half-hour event
        node : int
        path : list: chain from event node to node before candidate
        returns: data : time-shifted chain of data from candidate node, expanded by 50% on each side
        """
        time_difference = self._get_path_distance(path)
        start = self.query.index.min() - dt.timedelta(minutes=time_difference *
                                                      1.5)
        end = self.query.index.max() - dt.timedelta(minutes=time_difference *
                                                    0.5)
        data = _get_data(node, start, end)
        return data

    def _get_path_distance(self, path):
        return sum(self.graph[a][b]['distance']
                   for (a, b) in zip(path[:-1], path[1:]))

    def _check_node(self, node, path):
        if node == self.root:  # no need to check root
            return True, None
        node_data = self._get_time_shifted_data(node, path)
        if node_data is None:  # ie missing data
            if (pd.Timestamp.now() -
                    self.query.index.max()).total_seconds() / 60 / 60 < 8:
                logger.warning(
                    'Less than 8 hours have elapsed since event ended - data may not be final'
                )
                # TODO insert wait here
            return True, None
        logger.info(f'checking between {self.root} and {node}')
        return self.compare_data(node_data)

    def compare_data(self, node_data):
        """
        :param node_data: multi-dim data from potential suspect node
        :return: bool of suspect or not, plus array of scores for each sensor
        """
        # TODO look for matches historically and returning them to database
        # TODO add bank check to pinpointing func
        # TODO add comparison of flow, loads, and absolute values (won't be worse in query than in suspect)
        # TODO receive on which parameter alert was raised, and give that more weight
        # TODO talk to Naama about waiting for data
        # don't consider ORP/temp? When different from baseline
        # problem of matching on data which is mostly constant
        # need to think about what is unusual data
        errors = {}
        # intersection of root_data columns and node columns
        for parameter in self.query_parameters & set(node_data):
            if self.query[parameter].dtype != 'float64':
                logger.info(f'parameter {parameter} non-numeric, skipping')
                continue
            if all(node_data[parameter].isna()):
                logger.warning(f'all data for {parameter} is NaN...')
                continue
            logger.info(f'parameter={parameter}')
            norm_root, norm_node = self.normalised_query[
                parameter], _normalise(node_data[parameter])
            param_error = _get_dtw_distance(norm_root, norm_node)

            # normalise by dividing by root of length of series
            param_error /= len(norm_node)**0.5
            errors[parameter] = param_error
            if not np.isnan(param_error):
                logger.info(
                    f'{parameter} distance is {param_error}, (threshold={self.threshold})'
                )
        if len(errors) > 0:
            if np.nanmin(list(errors.values())) < self.threshold:
                return True, errors
        return False, None

    def depth_first_check(self, node, path):
        """
        DFS of tree from root. If node is suspect, keep searching children until end of chain
        :param node: node currently checking
        :param path: cumulative path from root
        :return: (what's important is not the return statement but the side affect of appending suspects)
        """
        is_suspect, score = self._check_node(node, path + [node])
        self.scores[node] = score
        if not is_suspect:  # if this isn't a suspect, append path from root to parent
            return False
        if True not in [
                self.depth_first_check(child, path + [node])
                for child in self.graph.successors(node)
        ]:
            self.suspects.append(Chain(path + [node]))
        return True

    def clean_up_chains(self):
        # clean trailing Nones, then since this may create duplicates, iterate over the
        # new suspects and remove duplicates and those which are contained in longer chains
        for chain in self.suspects:
            chain.nodes = chain.remove_trailing_nones(self.scores)
        # ie if not only Nones
        new_suspects = [chain for chain in self.suspects if chain.nodes]

        duplicated = []
        new_suspects.sort(key=lambda x: len(x.nodes))
        for idx, suspect1 in enumerate(new_suspects):
            for suspect2 in new_suspects[idx + 1:]:
                nodes1, nodes2 = suspect1.nodes, suspect2.nodes
                if len(nodes1) == len(nodes2):  # if they're duplicates
                    if nodes1 == nodes2:
                        duplicated.append(idx)
                if set(nodes1).issubset(nodes2):  # or subsets
                    duplicated.append(idx)
        self.suspects = [
            c for i, c in enumerate(new_suspects) if i not in set(duplicated)
        ]  # remove duplicates


def pinpoint(root, query, threshold):
    """
    :param root: int,
    :param query: a time sequence from a given node, with an area of interest starting at index until (index + duration)
    threshold of how far (DTW distance) we will consider a suspect
    :param threshold: DTW distance allowed for any one sensor
    :return: list of chains of suspects
    """
    # if query in bank:
    #     return 'query already exists in bank'
    # # TODO bank should use multidim DTW to check if signature already seen
    # bank.append(query)
    graph = get_graph(root)
    pinpointer = Pinpointer(root, graph, query, threshold)
    pinpointer.depth_first_check(root, [])
    pinpointer.clean_up_chains()
    ranked_suspects = sorted([(x.nodes, x.get_score(pinpointer.scores))
                              for x in pinpointer.suspects],
                             key=lambda x: x[1])
    return ranked_suspects


if __name__ == '__main__':
    root_node = 3316  # 3178
    # query = pd.read_csv('query.csv', index_col=0)
    orig_query = pd.read_csv('yoke.csv', index_col=0)
    orig_query.index = pd.to_datetime(orig_query.index)
    print(pinpoint(root_node, orig_query, threshold=0.35))
