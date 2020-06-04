import datetime as dt
import pickle

import pandas as pd
from tslearn.metrics import dtw_subsequence_path

from tools import get_data, get_graph_from_point

a = pd.date_range(pd.Timestamp('2020-05-30 07:00:00'), pd.Timestamp('2020-05-30 13:00:00'), freq='5min')
b = [1312., 1312., 1312., 1315.2, 1323.2, 1331.2, 1340.26666667, 1350.93333333, 1361.6, 1373.33333333, 6.66666667,
     1400., 1414.4, 1430.4, 1446.4, 1462.4, 1478.4, 1494.4, 1508.26666667, 1518.93333333, 1.6, 1539.2, 1547.2, 1555.2,
     1563.2, 1571.2, 1579.2, 1585.06666667, 1587.73333333, 1590.4, 1592., 1592., 1., 1592., 1592., 1592., 1589.86666667,
     1584.53333333, 1579.2, 1571.73333333, 1561.06666667, 1550.4, 1.73333333, 1529.06666667, 1518.4, 1507.73333333,
     1497.06666667, 1486.4, 1475.73333333, 1465.06666667, 1.4, 1444.8, 1436.8, 1428.8, 1420.8, 1412.8, 1404.8,
     1397.86666667, 1392.53333333, 1387.2, 1380.8, 1.8, 1364.8, 1358.93333333, 1356.26666667, 1353.6, 1350.93333333,
     1348.26666667, 1345.6, 1342.93333333, 1.26666667, 1337.6, 1333.86666667]

subseq = pd.Series(b, index=a)
event_node = 1332
wwtp_node = 1012
# graph = get_graph_from_point(wwtp_node)
with open('graph.pkl', 'rb') as f:
    graph = pickle.load(f)


def pinpointing(ts, index, duration, node, parameter, threshold=5):
    """
    input: sequence of points, indexed by time
    output: list of suspects
    BFS of the graph. At each level it checks for suspects above a set threshold. it then continues searching
    below them, and adding them to the list, with a confidence level based on closeness to original sequence
    if a point is blank (no data), it is automatically considered a suspect
    the search terminates when no suspects are above the level, or the tree is finished
    """
    candidates = list(graph.successors(node))
    if len(candidates) == 0:
        return [node]
    to_check = []
    start, end = ts.index.min(), ts.index.max()
    query = ts.iloc[index:index+duration].values
    for candidate in candidates:
        start += dt.timedelta(minutes=graph[node][candidate]['distance'])
        end += dt.timedelta(minutes=graph[node][candidate]['distance'])
        new_ts = get_data(candidate, start, end)[parameter].fillna(0).values
        dtw_distance = dtw_subsequence_path(query, new_ts)[1]
        if dtw_distance < threshold:
            to_check.append(candidate)
    return to_check


demo = pd.read_csv('foo.csv', header=None, index_col=0, parse_dates=True).iloc[:, 0]
a = pinpointing(demo, 370, 70, 1012, 'EC')
print(a)
