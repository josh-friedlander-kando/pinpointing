import numpy as np
import pandas as pd
import networkx as nx


def generate_data(start_time, end_time):
    # generate random (but realistic) EC/pH data between 2 dates)
    return pd.DataFrame({'EC': np.random.randint(950*10, 3000*10, 24*12) / 10,
                         'PH': np.random.randint(5.5*1000, 8*1000, 24*12) / 1000},
                        index=pd.date_range(start_time, end_time, freq='5min'))


class TestData:
    # basic idea here is to generate a fake graph and event, with timestamp EC/PH data, and check that we are finding it
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_weighted_edges_from([(2, 3, 60), (1, 2, 120), (3, 4, 15)])
        nx.set_node_attributes(self.graph, {1: 'one', 2: 'two', 3: 'three', 4: 'four'}, 'name')
        seed = 1
        start, end = pd.to_datetime('01/01/2020 00:00'), pd.to_datetime('01/01/2020 23:55')
        setattr(self, self.graph.nodes[1]['name'], generate_data(start, end))
        for (in_node, out_node) in sorted(self.graph.edges()):
            seed += 1
            np.random.seed(seed)
            start += pd.Timedelta(minutes=self.graph[in_node][out_node]['weight'])
            end += pd.Timedelta(minutes=self.graph[in_node][out_node]['weight'])
            setattr(self, self.graph.nodes[out_node]['name'], generate_data(start, end))


t = TestData()
print(t.one.head())
