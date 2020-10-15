import datetime as dt
import logging
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from kando import kando_client
from scipy.stats import zscore
from tslearn.metrics import dtw_subsequence_path

load_dotenv()

logging.basicConfig()

logger = logging.getLogger('pinpointing')
logger.setLevel(logging.DEBUG)

client = kando_client.client(os.getenv("KANDO_URL"), os.getenv("KEY"),
                             os.getenv("SECRET"))
# z_normaliser = TimeSeriesScalerMeanVariance()

# with open('graph.pkl', 'rb') as f:
#     graph = pickle.load(f)


def _parser(node, graph):
    if len(node['children']) == 0:
        graph.add_node(node['point_id'])
        return
    for child in node['children']:
        graph.add_edges_from([(node['point_id'], child['point_id'])],
                             distance=child['parent_distance'])
        _parser(child, graph)


def get_graph_from_point(point):
    raw = client.network_graph(point)
    g = nx.DiGraph()
    _parser(raw, g)
    return g


def _get_data(point, start, end):
    # convert time to epoch, fetch from API, turn into dataframe, return
    return_data = client.get_data(point_id=point,
                                  start=start.timestamp(),
                                  end=end.timestamp())
    samples = return_data['samplings']
    if len(samples) == 0:
        logger.info(f'No data at all for node {point}, continuing the search')
        return None
    # TODO more complex - must check when last time responded. If under 12(?) hours, just return a "wait and see"
    df = pd.DataFrame(samples).T
    df.index = pd.to_datetime(df.index, unit='s')
    df = (df.drop(columns=['DateTime']).astype(float).resample(
        '1min').interpolate().resample('5min').mean())
    return df


def _get_dtw_distance(query, ts):
    return dtw_subsequence_path(query, ts)[1]


def _normalise(arr):
    return zscore(arr)


class PinpointHelper:
    def __init__(self, graph, threshold):
        self.graph = graph
        self.threshold = threshold
        # self.z_normaliser = zscore()

    def _get_time_shifted_data(self, chain, candidate, query):
        """
        Given a chain and a node, and (start,end) times, find the equivalent time-shifted (start,end) for the node
        Note that we "pad" the time-shifting by 50% each way
        So for example if event takes place at 08:00-08:30 at event node, and candidate node is 3 hours behind,
        we will return data from start - (distance * 1.5), ie 03:30 until end - (distance * 0.5) ie 07:00
        Within this 3.5 hour space we will search for a pattern matching the half-hour event
        chain : list: Chain from event node to node before candidate
        candidate : int
        query : ts, possibly multi-dimensional
        returns: data : chain of data from candidate matching pattern, expanded by 50% on each side
        """
        time_difference = self._get_chain_distance(chain + [candidate])
        start = query.index.min() - dt.timedelta(minutes=time_difference * 1.5)
        end = query.index.max() - dt.timedelta(minutes=time_difference * 0.5)
        data = _get_data(candidate, start, end)
        return data

    def _get_chain_distance(self, chain):
        return sum(self.graph[a][b]['distance']
                   for (a, b) in zip(chain[:-1], chain[1:]))

    def check_child(self, chain, child, query):
        child_data = self._get_time_shifted_data(chain, child, query)
        if child_data is None:  # ie missing data
            return True, None
        logger.info(f'checking between {chain[0]} and {child}')
        return self.compare_data(query, child_data)

    def compare_data(self, root_data, node_data):
        """

        :param root_data: multi-dim query data
        :param node_data: multi-dim data from potential suspect node
        :return: bool of suspect or not, plus array of scores for each sensor
        """
        # TODO look for matches historically and returning them to database
        # TODO return scores of chains
        # TODO add comparison of flow, loads, and absolute values (won't be worse in query than in suspect)
        # TODO if 2 matches better than just one
        # TODO receive on which parameter alert was raised, and give that more weight
        # TODO talk to Naama about waiting for data
        # don't consider ORP/temp? When different from baseline
        # problem of matching on data which is mostly constant
        # need to think about what is unusual data
        errors = []
        # intersection of root_data columns and child columns
        for parameter in set(root_data) & set(node_data):
            if root_data[parameter].dtype != 'float64':
                logger.info(f'parameter {parameter} non-numeric, skipping')
                continue
            if all(node_data[parameter].isna()):
                logger.warning(f'all data for {parameter} is NaN...')
                continue
            logger.info(f'parameter={parameter}')
            norm_root_ts = _normalise(root_data[parameter])
            norm_child_ts = _normalise(node_data[parameter])
            param_error = _get_dtw_distance(norm_root_ts, norm_child_ts)
            # normalise by dividing by root of length of series
            param_error /= len(norm_child_ts)**0.5
            errors.append(param_error)
            if not np.isnan(param_error):
                logger.info(
                    f'{parameter} distance is {param_error}, (threshold={self.threshold})'
                )
        errors = np.array(errors)
        if np.nanmin(errors) < self.threshold:
            return True, errors[errors < self.threshold]
        return False, None
