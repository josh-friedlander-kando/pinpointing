import datetime as dt
import os
import pickle

import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from kando import kando_client
from tslearn.metrics import dtw_subsequence_path
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

load_dotenv()
client = kando_client.client(os.getenv("KANDO_URL"), os.getenv("KEY"), os.getenv("SECRET"))
z_normaliser = TimeSeriesScalerMeanVariance()

with open('graph.pkl', 'rb') as f:
    graph = pickle.load(f)


def _parser(node, graph):
    if len(node['children']) == 0:
        graph.add_node(node['point_id'])
        return
    for child in node['children']:
        graph.add_edges_from([(node['point_id'], child['point_id'])], distance=child['parent_distance'])
        _parser(child, graph)


def get_graph_from_point(point):
    raw = client.network_graph(point)
    g = nx.DiGraph()
    _parser(raw, g)
    return g


def _get_data(point, start, end):
    # convert time to epoch, fetch from API, turn into dataframe, return
    return_data = client.get_data(point_id=point, start=start.timestamp(), end=end.timestamp())
    if return_data['point']['device']['unit_id'] is None:
        return None
    samples = return_data['samplings']
    if len(samples) == 0:
        return None
    # TODO more complicated - must check when last time responded. If under 12(?) hours, just return a "wait and see"
    df = pd.DataFrame(samples).T
    df.index = pd.to_datetime(df.index, unit='s')
    df = (df.drop(columns=['DateTime'])
          .astype(float)
          .resample('1min')
          .interpolate()
          .resample('5min')
          .mean())
    return df


def _get_dtw_distance(query, ts):
    return dtw_subsequence_path(query, ts)[1]


def _get_time_shifted_data(chain, candidate, query):
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
    time_difference = _get_chain_distance(chain + [candidate])
    start = query.index.min() - dt.timedelta(minutes=time_difference * 1.5)
    end = query.index.max() - dt.timedelta(minutes=time_difference * 0.5)
    data = _get_data(candidate, start, end)
    return data


def _get_chain_distance(chain):
    return sum(graph[a][b]['distance'] for (a, b) in zip(chain[:-1], chain[1:]))


def check_child(chain, child, query, threshold):
    error = 0
    child_data = _get_time_shifted_data(chain, child, query)
    if child_data is None:
        return -1
    for parameter in set(query) & set(child_data):  # intersection of query columns and child columns
        if all(child_data[parameter].isna()):
            return -1
        child_ts = z_normaliser.fit_transform(child_data[parameter]).reshape(-1)
        error += _get_dtw_distance(query[parameter], child_ts) < threshold
        error /= len(child_ts)**0.5  # normalise by dividing by root of length of series
    return error
