import pickle
import pandas as pd
from tools import check_child

with open('graph.pkl', 'rb') as f:
    graph = pickle.load(f)


def pinpointing(ts, index, duration, node, threshold=5):
    """
    input: a time sequence from a given node, with an area of interest starting at index until (index + duration)
    threshold of how far (DTW distance) we will consider a suspect
    output: list of chains of suspects
    BFS of the graph. At each level it checks for suspects above a set threshold. it then continues searching
    below them, and adding them to the list, with a confidence level based on closeness to original sequence
    if a point is blank (no data), it is automatically considered a suspect
    the search terminates when no suspects are above the level, or the tree is finished
    """
    query = ts.iloc[index:index + duration]
    level = 0
    suspects = []
    active_chains = [[node]]
    ts_start, ts_end = ts.index.min(), ts.index.max()
    while len(active_chains) > 0:
        new_active_chains = []
        for chain in active_chains:
            parent = chain[level]
            children = list(graph.successors(parent))
            # if node has no children who are suspects, it is end of chain, and so suspect
            # if it has at least one child below threshold, it is not end of chain
            end_of_chain = True
            for child in children:
                if check_child(chain, child, query, ts_start, ts_end, threshold):
                    new_active_chains.append(chain + [child])
                    end_of_chain = False
            if end_of_chain:
                suspects.append(chain)
        level += 1
        active_chains = new_active_chains
    return suspects


if __name__ == '__main__':
    demo = pd.read_csv('query.csv', index_col=0)
    demo.index = pd.to_datetime(demo.index)
    print(pinpointing(demo, 0, 10, 1012, threshold=50))
