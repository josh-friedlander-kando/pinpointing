from pinpointing.pinpointing import *


def test_data_fetcher():
    assert clean_fetch(1578, None, None) is None


def test_normalise():
    assert all(normalise(np.array([10, 20, 30])) == np.array([-1.224744871391589,  0,  1.224744871391589]))
    assert all(normalise(np.array([0, 0, 0])) == [0, 0, 0])
    res = normalise(np.array([np.NaN, np.NaN, np.NaN]))
    assert all(np.isnan(res))


test_data_fetcher()
test_normalise()
