from pinpointing.data_fetcher import KandoDataFetcher


def test_data_fetcher():
    client = KandoDataFetcher()
    assert client.get_pandas_data(1578, None, None) is None


test_data_fetcher()
