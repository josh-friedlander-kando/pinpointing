import datetime as dt
import logging
import os

import networkx as nx
import numpy as np
import pandas as pd
import pytz
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
    'PI': 0.9,
    'TSS': 0.5,
    'COD': 0.9,
    # 'ORP': 0.5,
    # 'TEMPERATURE': 0.8
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
    logger.info(f'getting data for {point} from {start} to {end}')
    df = pd.DataFrame(samples).T
    df.index = pd.to_datetime(
        df.index, unit='s').tz_localize('UTC').tz_convert('Asia/Jerusalem')
    df = (df.drop(columns=['DateTime']).astype(float).resample('5min').first())
    return df


def _get_dtw_distance(query, ts):
    return dtw_subsequence_path(query, ts)[1]


def _normalise(arr):
    return zscore(arr, nan_policy='omit')


class Chain:
    # initiated with a list of one or multiple nodes
    def __init__(self, nodes):
        self.nodes = nodes

    def add(self, node):
        self.nodes.append(node)

    def get_score(self, scores, threshold):
        """
        Returns weighted average of non-NaN/non-None values from sensors, per weights (above)
        For each node, we calculate the distance from the root, and the complement is normalised
        to a percentage between 0 and the threshold
        This percentage represents the confidence in this node being the source of the query
        :param scores: dict of scores for each sensor of each node
        :param threshold: max allowable score for a sensor to be considered
        :return: weighted score of this node
        """
        res = {}
        for node in self.nodes[1:]:
            if scores[node] is None:
                res[node] = 'Missing data'
                continue
            res[node] = {
                sensor: '{:.1%}'.format(((threshold - distance) / threshold) *
                                        scoring_weights[sensor])
                for sensor, distance in scores[node].items()
                if distance < threshold
            }
        return res

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
        self.threshold = threshold
        self.suspects = []
        self.scores = {}
        self.max_PI = query['PI'].max()

        query = query[scoring_weights].copy()  # drop what's not in weights
        for param in query:
            if query[param].dtype == 'float64':
                if not all(query[param].isna()):
                    query[param] = _normalise(query[param])

        self.query = query

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
            if (pd.Timestamp.now(pytz.FixedOffset(120)) -
                    self.query.index.max()).total_seconds() / 60 / 60 < 8:
                logger.warning(
                    'Less than 8 hours have elapsed since event ended - data may not be final'
                )
            return True, None
            # TODO talk to Naama about waiting for data
        logger.info(f'checking between {self.root} and {node}')
        return self.compare_data(node_data)

    def compare_data(self, node_data):
        """
        :param node_data: multi-dim data from potential suspect node
        :return: bool of suspect or not, plus array of scores for each sensor
        """
        # TODO look for matches historically and returning them to database
        # and then add bank check to pinpointing func
        # TODO receive on which parameter alert was raised, and give that more weight
        if node_data['PI'].max() < self.max_PI:
            logger.info('node rejected because of lower PI')
            return False, None
        errors = {}
        # intersection of root_data columns and node columns
        for parameter in set(
                self.query) & set(node_data) & set(scoring_weights):
            if self.query[parameter].dtype != 'float64':
                logger.info(f'parameter {parameter} non-numeric, skipping')
                continue
            if all(node_data[parameter].isna()):
                logger.warning(f'all data for {parameter} is NaN...')
                continue
            logger.info(f'parameter={parameter}')

            norm_root, norm_node = self.query[parameter].dropna(), _normalise(
                node_data[parameter].dropna())
            param_error = _get_dtw_distance(norm_root, norm_node)

            # take the mean
            param_error /= len(norm_node)
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
    return [
        x.get_score(pinpointer.scores, pinpointer.threshold)
        for x in pinpointer.suspects
    ]


if __name__ == '__main__':
    # root_node = 3316  # 3178
    root_node = 1012
    # orig_query = pd.read_csv('yoke.csv', index_col=0)
    orig_query = pd.read_csv('soreq.csv', index_col=0)
    orig_query.index = pd.to_datetime(orig_query.index)
    print(pinpoint(root_node, orig_query, threshold=0.06))
