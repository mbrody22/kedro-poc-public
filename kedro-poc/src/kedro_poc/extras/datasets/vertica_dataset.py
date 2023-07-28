from kedro.io import AbstractDataSet
from vertica_python import connect
import pandas as pd
from mmpac import vertica_setup, get_query, read_sql, vertica_disconnect


class VerticaSQLQueryDataSet(AbstractDataSet):
    def __init__(self, credentials, filepath):
        self.server = credentials['server']
        self.user = credentials['user']
        self.password = credentials['password']
        self.filepath = filepath

    def _load(self) -> pd.DataFrame:
        # Connect to vertica using mmpac
        vertica_setup(self.server, self.user, self.password, connection_timeout=240)
        # Retrieve the data
        df = get_query(sql=read_sql(self.filepath))
        # Disconnect from Vertica
        vertica_disconnect()
        return df

    def _save(self, data: pd.DataFrame) -> None:
        # Implement the saving logic if needed
        pass

    def _describe(self):
        return dict(
            server=self.server,
            query=self.filepath
        )
