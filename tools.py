import os
import pandas as pd
from kando import kando_client
import networkx as nx
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("KANDO_URL")
key = os.getenv("KEY")
secret = os.getenv("SECRET")
client = kando_client.client(url, key, secret)


def parser(node, graph):
    if len(node['children']) == 0:
        graph.add_node(node['point_id'])
        return
    for child in node['children']:
        graph.add_edges_from([(node['point_id'], child['point_id'])], distance=child['parent_distance'])
        parser(child, graph)


def get_graph_from_point(point):
    raw = client.network_graph(point)
    G = nx.DiGraph()
    parser(raw, G)
    return G


def get_data(point, start, end):
    # convert time to epoch, fetch from API, turn into dataframe, return
    df = pd.DataFrame(client.get_data(
        point_id=point, start=start.timestamp(), end=end.timestamp())['samplings']).T
    df['point_id'] = point
    df.index = pd.to_datetime(df.index, unit='s')
    df = df.resample('1min').interpolate().resample('5min').mean()
    return df
