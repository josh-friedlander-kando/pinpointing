from datetime import timedelta
import logging
import networkx as nx
import numpy as np
import pandas as pd
import pytz
from scipy.stats import zscore
from tslearn.metrics import dtw_subsequence_path

logging.basicConfig()

logger = logging.getLogger("pinpointing")
logger.setLevel(logging.DEBUG)

scoring_weights = {
    "EC": 1,
    "PH": 1,
    "PI": 0.9,
    "TSS": 0.5,
    "COD": 0.9,
    # 'ORP': 0.5,
    # 'TEMPERATURE': 0.8
}


def _get_dtw_distance(query, ts):
    return dtw_subsequence_path(query, ts)[1]


def normalise(arr):
    if (all(np.isnan(arr))) | (all(arr == 0)):
        return arr
    return zscore(arr, nan_policy="omit")


class Chain:
    """
    Chain of suspects with methods to clear trailing nones and calculate chain score.
    Initialised with a list of one or more nodes.
    """
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
                res[node] = "Missing data"
                continue
            res[node] = {
                sensor: "{:.1%}".format(((threshold - distance) / threshold) *
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


class KandoTree:
    """
    Object that provides the graph/tree traversal functionality of this library.
    Builds and stores the attribute nx_graph which is a NetworkX graph built from the
    Kando API network_graph response for a root node, containing its ancestors/descendants.
    """
    def __init__(self, client, point):
        self.tree_details = {}
        root_data = client.network_graph(point)
        nx_graph = nx.DiGraph()
        self._parser(root_data, nx_graph)

        self._parse_response(root_data)
        nx.set_node_attributes(nx_graph, self.tree_details)
        ancestors = [
            x["point"]["id"] for x in root_data["ancestors"]
            if x["point"]["device"]["unit_id"] is not None
        ]
        distances = [root_data["parent_distance"]] + [
            x["parent_distance"] for x in root_data["ancestors"]
            if x["point"]["device"]["unit_id"] is not None
        ][:-1]
        self.ancestors = list(zip(ancestors, distances))
        self.nx_graph = nx_graph

    def _parser(self, node, graph):
        if len(node["children"]) == 0:
            graph.add_node(node["point_id"])
            return
        for child in node["children"]:
            graph.add_edges_from([(node["point_id"], child["point_id"])],
                                 weight=child["parent_distance"])
            self._parser(child, graph)

    def _parse_response(self, response):
        """
        Parse API graph structure for "name" and "has_device"
        :param response:
        :return: nothing, but populates self.tree_details with data
        """
        self.tree_details[response["point_id"]] = {
            "name": response["point"]["name"],
            "has_device": response["point"]["device"]["unit_id"] is not None,
        }
        if len(response["children"]) > 0:
            for ch in response["children"]:
                self._parse_response(ch)

    def get_device_dict(self):
        return nx.get_node_attributes(self.nx_graph, "has_device")


class Pinpointer:
    """
    The Pinpointer object does most of the work. It initializes the KandoTree, fetches the
    query data, and traverses the graph upstream or downstream, fetching time-shifts data
    and comparing it. It contains suspects, which are Chain objects.
    """
    def __init__(self, client, params):
        self.scores = {}
        self.suspects = []
        self.root = params["root"]
        self.data_fetcher = client
        self.graph = KandoTree(self.data_fetcher, params["root"])
        self.threshold = params["threshold"]
        self.device_dict = self.graph.get_device_dict()

        query = self.data_fetcher.get_pandas_data(params["root"],
                                                  params["start"],
                                                  params["end"])

        # drop what's not in weights
        query = query[set(scoring_weights).intersection(query.columns)].copy()
        for param in query:
            if query[param].dtype == "float64":
                query[param] = normalise(query[param])

        self.query = query
        self.max_PI = query["PI"].max()
        if "sensor" in params:
            self.event_sensor = params["sensor"]

    def _get_time_shifted_data(self, node, path=None, time_difference=None):
        """
        Given a chain and a node, and (start,end) times, find the equivalent time-shifted (start,end) for the node
        Note that we "pad" the time-shifting by 50% each way
        So for example if event takes place at 08:00-08:30 at event node, and candidate node is 3 hours behind,
        we will return data from start - (distance * 1.5), ie 03:30 until end - (distance * 0.5) ie 07:00
        Within this 3.5 hour space we will search for a pattern matching the half-hour event
        This example is for the common (upstream) case. When downstream we shift time forward.
        node : int
        path : list: chain from event node to node before candidate
        returns: data : time-shifted chain of data from candidate node, expanded by 50% on each side
        """
        if path is None:
            assert (
                time_difference
                is not None), "Either path or time_distance must be provided"
            start = self.query.index.min() + timedelta(
                minutes=time_difference * 0.5)
            end = self.query.index.max() + timedelta(minutes=time_difference *
                                                     1.5)

        else:
            time_difference = self._get_path_distance(path)
            start = self.query.index.min() - timedelta(
                minutes=time_difference * 1.5)
            end = self.query.index.max() - timedelta(minutes=time_difference *
                                                     0.5)
        return self.data_fetcher.get_pandas_data(node, start, end)

    def _get_path_distance(self, path):
        return sum(self.graph.nx_graph[a][b]["weight"]
                   for (a, b) in zip(path[:-1], path[1:]))

    def _check_node(self, node, path):
        if node == self.root:  # no need to check root
            return True, None
        if not self.device_dict[node]:
            return True, None
        node_data = self._get_time_shifted_data(node, path=path)
        if node_data is None:  # ie missing data
            # TODO should make TZ here and in KandoDataGetter configurable!
            if (pd.Timestamp.now('UTC') -
                    self.query.index.max()).total_seconds() / 60 / 60 < 8:
                logger.warning(
                    "Less than 8 hours have elapsed since event ended - data may not be final"
                )
            return True, None
            # TODO talk to Naama about waiting for data
        logger.info(f"checking between {self.root} and {node}")
        return self.compare_data(node_data)

    def compare_data(self, node_data):
        """
        :param node_data: multi-dim data from potential suspect node
        :return: bool of suspect or not, plus array of scores for each sensor
        """
        # TODO look for matches historically and returning them to database
        # and then add bank check to pinpointing func
        # TODO receive on which parameter alert was raised, and give that more weight
        # if node_data['PI'].max() < self.max_PI:
        #     logger.info('node rejected because of lower PI')
        #     return False, None
        errors = {}
        # intersection of root_data columns and node columns
        for parameter in set(
                self.query) & set(node_data) & set(scoring_weights):
            if self.query[parameter].dtype != "float64":
                logger.info(f"parameter {parameter} non-numeric, skipping")
                continue
            if all(node_data[parameter].isna()):
                logger.warning(f"all data for {parameter} is NaN...")
                continue
            logger.info(f"parameter={parameter}")

            norm_root, norm_node = self.query[parameter].dropna(), normalise(
                node_data[parameter].dropna())
            param_error = _get_dtw_distance(norm_root, norm_node)

            # take the mean
            param_error /= len(norm_node)
            errors[parameter] = param_error
            if not np.isnan(param_error):
                logger.info(
                    f"{parameter} distance is {param_error}, (threshold={self.threshold})"
                )
        if len(errors) > 0:
            if np.nanmin(list(errors.values())) < self.threshold:
                return True, errors
        return False, None

    def search_downstream(self):
        distance = 0
        for node, node_dist in self.graph.ancestors:
            distance += node_dist
            logger.info(f"checking downstream from {self.root} to {node}")
            node_data = self._get_time_shifted_data(node,
                                                    time_difference=distance)
            if node_data is None:  # ie missing data
                if (pd.Timestamp.now(pytz.FixedOffset(120)) -
                        self.query.index.max()).total_seconds() / 60 / 60 < 8:
                    logger.warning(
                        "Under 8h elapsed since event end, data may not be final"
                    )
                continue
                # TODO talk to Naama about waiting for data
            is_suspect, score = self.compare_data(node_data)
            if not is_suspect:
                return
            self.suspects.append((node, score))
        return self.suspects

    def search_upstream(self, node, path):
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
                self.search_upstream(child, path + [node])
                for child in self.graph.nx_graph.successors(node)
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
