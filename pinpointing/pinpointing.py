from datetime import datetime
import logging
from tools import Pinpointer
from data_fetcher import KandoDataFetcher

logging.basicConfig()

logger = logging.getLogger('pinpointing')
logger.setLevel(logging.DEBUG)


def pinpoint(params, direction='up'):
    """
    param is a dict containing:
    root: int,
    query: a time sequence from a given node, with an area of interest starting at index until (index + duration)
    threshold of how far (DTW distance) we will consider a suspect
    threshold: DTW distance allowed for any one sensor
    sensor: sensor for which the event was flagged
    :return: list of chains of suspects
    """
    # bank = []
    # if query in bank:
    #     return 'query already exists in bank'
    # bank.append(query)
    client = KandoDataFetcher()
    pinpointer = Pinpointer(client, params)
    if direction == 'down':
        return pinpointer.search_downstream()

    pinpointer.search_upstream(params['root'], [])
    pinpointer.clean_up_chains()
    return [
        x.get_score(pinpointer.scores, pinpointer.threshold)
        for x in pinpointer.suspects
    ]


if __name__ == '__main__':
    # root_node = 1191
    root_node = 1012
    start = (2020, 10, 12)
    end = (2020, 10, 15, 12)
    logger.info(
        f'Query start date is {datetime.strftime(datetime(*start), "%d %B %Y %H:%M")}'
    )
    logger.info(
        f'Query end date is {datetime.strftime(datetime(*end), "%d %B %Y %H:%M")} \n'
    )

    pinpoint_params = {
        'root': root_node,
        'start': start,
        'end': end,
        'threshold': 0.35,
        'sensor': 'PI'
    }
    print(pinpoint(pinpoint_params, 'up'))
