import datetime as dt
import pickle
import os
import pandas as pd
from kando import kando_client
import networkx as nx
from dotenv import load_dotenv
from tslearn.metrics import dtw_subsequence_path

load_dotenv()
url = os.getenv("KANDO_URL")
key = os.getenv("KEY")
secret = os.getenv("SECRET")
client = kando_client.client(url, key, secret)

with open('graph.pkl', 'rb') as f:
    graph = pickle.load(f)
parameter = 'EC'


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
    df = pd.DataFrame(client.get_data(
        point_id=point, start=start.timestamp(), end=end.timestamp())['samplings']).T
    df['point_id'] = point
    df.index = pd.to_datetime(df.index, unit='s')
    df = df.resample('1min').interpolate().resample('5min').mean()
    return df


def _get_dtw_distance(query, ts):
    return dtw_subsequence_path(query, ts)[1]


def _get_time_shifted_data(chain, candidate, start, end):
    """
    Given tail and head nodes, and a ts from the tail, find the equivalent ts for the head time-shifted by distance
    between the two in minutes
    """
    time_difference = _get_chain_distance(chain + [candidate])
    start += dt.timedelta(minutes=time_difference)
    end += dt.timedelta(minutes=time_difference)
    data = _get_data(candidate, start, end)
    if parameter not in data:
        return None
    return data[parameter].fillna(0).values


def _get_chain_distance(chain):
    return sum(graph[a][b]['distance'] for (a, b) in zip(chain[:-1], chain[1:]))


def check_child(chain, child, query, start, end, threshold):
    new_ts = _get_time_shifted_data(chain, child, start, end)
    if new_ts is None:
        return True
    return _get_dtw_distance(query, new_ts) < threshold


# a = _get_data(1560, pd.Timestamp(1590717600), pd.Timestamp(1591003200))
# print('a')