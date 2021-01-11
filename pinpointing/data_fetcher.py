from datetime import datetime
import logging
import os
import pandas as pd
from dotenv import load_dotenv
from kando.kando_client import KandoClient

load_dotenv()

logging.basicConfig()

logger = logging.getLogger("pinpointing")
logger.setLevel(logging.DEBUG)


class KandoDataFetcher(KandoClient):
    """
    KandoDataFetcher subclasses the KandoClient and adds the get_pandas_data method,
    which accepts a broader range of DateTime objects, and returns a Pandas dataframe
    """
    def __init__(self):
        host, key, secret = (os.getenv("KANDO_URL"), os.getenv("KEY"),
                             os.getenv("SECRET"))
        super().__init__(host, key, secret)

    def get_pandas_data(self, point_id, start=None, end=None):
        """
        Wrapper for the get_all method with nice defaults. By default returns last six months.
        point_id: int, point to search
        start: (int, int, int[, int[, int[, int]]]) - date in form (year, month, day[, hour[, minute[, second]]])
        end: (int, int, int[, int[, int[, int]]]) - date in form (year, month, day[, hour[, minute[, second]]])
        :return: a Pandas dataframe with relevant data
        """
        if start is None or end is None:
            logger.error("Dates not provided. No data provided.")
            return None
        if type(start) != pd.Timestamp:
            try:
                start, end = datetime(*start), datetime(*end)
            except ValueError:
                logger.error(
                    "invalid dates. Must be in the form (1,1,2020),(15,6,2020)..."
                )
                return None

        data = self.get_all(point_id=point_id,
                            start=start.timestamp(),
                            end=end.timestamp())
        if len(data["samplings"]) == 0:
            print(f"No data found for point {point_id} at selected dates")
            return None

        df = pd.DataFrame(data["samplings"]).T

        # convert datetime, localise it as UTC (meaning it is already
        # but just ensure that Pandas treats it as that
        df.index = (pd.to_datetime(df.index, unit="s").tz_localize("UTC"))
        df = df.drop(columns=["DateTime"]).astype(float)
        return df
