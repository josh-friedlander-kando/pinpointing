import pandas as pd
from tools import PinpointHelper, get_graph_from_point

bank = []


def pinpointing(query, node, tools, threshold=0.7):
    """
    input: a time sequence from a given node, with an area of interest starting at index until (index + duration)
    threshold of how far (DTW distance) we will consider a suspect
    output: list of chains of suspects
    BFS of the graph. At each level it checks for suspects above a set threshold. it then continues searching
    below them, and adding them to the list, with a confidence level based on closeness to original sequence
    if a point is blank (no data), it is automatically considered a suspect
    the search terminates when no suspects are above the level, or the tree is finished
    """
    if query in bank:
        return 'query already exists in bank'
    bank.append(query)  # TODO bank should use multidim DTW to check if signature already seen
    level = 0
    suspects = []
    active_chains = [[node]]
    while len(active_chains) > 0:
        new_active_chains = []
        for chain in active_chains:
            parent = chain[level]
            children = list(tools.graph.successors(parent))
            # if node has no children who are suspects, it is end of chain, and so suspect
            # if it has at least one child below threshold, it is not end of chain
            end_of_chain = True
            for child in children:
                if tools.check_child(chain, child, query, threshold):
                    new_active_chains.append(chain + [child])
                    end_of_chain = False
            if end_of_chain:
                suspects.append(chain)
        level += 1
        active_chains = new_active_chains
    return suspects


if __name__ == '__main__':
    root_node = 3178
    demo = pd.read_csv('query.csv', index_col=0)
    demo.index = pd.to_datetime(demo.index)
    helper = PinpointHelper(get_graph_from_point(root_node), threshold=0.4)
    print(pinpointing(demo, root_node, helper))
