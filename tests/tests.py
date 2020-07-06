from pinpointing import pinpointing
from testdata import TestData
from tools import _get_dtw_distance, compare_data


t = TestData()
print(t.one.head())
print(t.two.head())
compare_data(t.one, t.two, 0.4)
# print(_get_dtw_distance([1, 2, 3], [56, 34, 12, 1, 2, 2, 2, 3, 4]))


