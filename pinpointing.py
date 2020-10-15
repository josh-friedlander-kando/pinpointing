import pandas as pd
import datetime as dt
import logging
import os
import pickle

import networkx as nx
import numpy as np
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

# with open('graph.pkl', 'rb') as f:
#     graph = pickle.load(f)


def _parser(node, graph):
    if len(node['children']) == 0:
        graph.add_node(node['point_id'], name=node['point']['name'])
        return
    for child in node['children']:
        graph.add_edges_from([(node['point_id'], child['point_id'])],
                             distance=child['parent_distance'])
        _parser(child, graph)


def get_graph(point):
    g = client.network_graph(point)
    G = nx.DiGraph()
    _parser(g, G)
    return G


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


class Chain:
    # initiated with a list of one or multiple nodes
    def __init__(self, initial_nodes):
        self.nodes = initial_nodes

    def add(self, node):
        self.nodes.append(node)

    def get_end_of_chain(self):
        return self.nodes[-1]

    # TODO get average scores, etc


class Pinpointer:
    def __init__(self, root, graph, query, threshold):
        self.root = root
        self.graph = graph
        self.query = query
        self.threshold = threshold
        self.suspects = []

    def _get_time_shifted_data(self, node, path):
        """
        Given a chain and a node, and (start,end) times, find the equivalent time-shifted (start,end) for the node
        Note that we "pad" the time-shifting by 50% each way
        So for example if event takes place at 08:00-08:30 at event node, and candidate node is 3 hours behind,
        we will return data from start - (distance * 1.5), ie 03:30 until end - (distance * 0.5) ie 07:00
        Within this 3.5 hour space we will search for a pattern matching the half-hour event
        node : int
        path : list: Chain from event node to node before candidate
        returns: data : time-shifted chain of data from candidate node, expanded by 50% on each side
        """
        time_difference = self._get_path_distance(path)
        start = self.query.index.min() - dt.timedelta(minutes=time_difference * 1.5)
        end = self.query.index.max() - dt.timedelta(minutes=time_difference * 0.5)
        data = _get_data(node, start, end)
        return data

    def _get_path_distance(self, path):
        return sum(self.graph[a][b]['distance'] for (a, b) in zip(path[:-1], path[1:]))
        # return nx.shortest_path_length(self.graph, self.root, node, weight='weight')

    def _check_node(self, node, path):
        node_data = self._get_time_shifted_data(node, path)
        if node_data is None:  # ie missing data
            return True, None
        logger.info(f'checking between {self.root} and {node}')
        return self.compare_data(node_data)

    def compare_data(self, node_data):
        """
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
        for parameter in set(self.query) & set(node_data):
            if self.query[parameter].dtype != 'float64':
                logger.info(f'parameter {parameter} non-numeric, skipping')
                continue
            if all(node_data[parameter].isna()):
                logger.warning(f'all data for {parameter} is NaN...')
                continue
            logger.info(f'parameter={parameter}')
            norm_root_ts = _normalise(self.query[parameter])
            norm_child_ts = _normalise(node_data[parameter])
            param_error = _get_dtw_distance(norm_root_ts, norm_child_ts)
            # normalise by dividing by root of length of series
            param_error /= len(norm_child_ts)**0.5
            errors.append(param_error)
            if not np.isnan(param_error):
                logger.info( f'{parameter} distance is {param_error}, (threshold={self.threshold})' )
        errors = np.array(errors)
        if np.nanmin(errors) < self.threshold:
            return True, errors[errors < self.threshold]
        return False, None

    def depth_first_check(self, node, path):
        """
        DFS of tree from root. If node is suspect, keep searching children until end of chain
        :param node: node currently checking
        :param path: cumulative path from root
        :return: (what's important is not the return statement but the side affect of appending suspects)
        """
        is_suspect, score = self._check_node(node, path + [node])
        if not is_suspect:  # if this isn't a suspect, append path from root to parent
            return False
        if True not in [self.depth_first_check(child, path + [node]) for child in self.graph.successors(node)]:
            self.suspects.append(Chain(path + [node]))
        return True


def pinpoint(root, query, threshold):
    """
    :param root: int,
    :param query: a time sequence from a given node, with an area of interest starting at index until (index + duration)
    threshold of how far (DTW distance) we will consider a suspect
    :param threshold: DTW distance allowed for any one sensor
    :return: list of chains of suspects
    """
    graph = get_graph(root)
    pinpointer = Pinpointer(root, graph, query, threshold)
    pinpointer.depth_first_check(root, [])
    return [x.nodes for x in pinpointer.suspects]

    # if query in bank:
    #     return 'query already exists in bank'
    # # TODO bank should use multidim DTW to check if signature already seen
    # bank.append(query)
    # level = 0
    # suspects = []
    # active_chains = [Chain(node)]
    # while any([c.active for c in active_chains]):
    #     # pinpointing ends when no active chains
    #     new_active_chains, chain_score = [], 0
    #     for chain in active_chains:
    #         parent = chain.get_end_of_chain()
    #         children = list(tools.graph.successors(parent))
    #         # if node has no children who are suspects, it is end of chain, and so suspect
    #             # if it has at least one child below threshold, it is not end of chain
    #             # end_of_chain = True  not needed now since activation state = True?
    #         chain.active = False
    #         for child in children:
    #             is_suspect, score = tools.check_child(chain.nodes, child, self.query)
    #             if is_suspect:
    #                 chain.add(child)
    #                 chain.active = True
    #                 new_active_chains.append(chain)
    #                 # if score is not None:
    #                 #     chain_score += min(score)
    #         if not chain.active:
    #             suspects.append(chain)
    #             # print(f'Score for chain {chain} is {score}')
    #     level += 1
    #     active_chains = new_active_chains
    # return suspects


if __name__ == '__main__':
    # root_node = 3178
    root_node = 3316
    # demo = pd.read_csv('query.csv', index_col=0)
    demo = pd.read_csv('yokn.csv', index_col=0)
    demo.index = pd.to_datetime(demo.index)
    print(pinpoint(root_node, demo, threshold=0.35))
    # print(pinpointing(demo, root_node, helper))
