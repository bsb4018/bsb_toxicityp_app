import sys
import numpy as np
import pandas as pd
from toxicpred.exception import ToxicityException
from toxicpred.configuration.cassandra_connection import AstraCassandraConfig
from astrapy.rest import http_methods

class ToxicityData:
    """
    Export data from astra cassandra database to pandas dataframe
    """
    def __init__(self):
        try:
            self.astra_cassandra_configurer = AstraCassandraConfig()
        except Exception as e:
            raise ToxicityException(e, sys)


    def export_from_astra_database_to_dataframe_using_restapi(self) -> pd.DataFrame:
        try:
            astra_http_client = self.astra_cassandra_configurer.getAstraHTTPClient()
            url = f"/api/rest/v2/keyspaces/{self.astra_cassandra_configurer.keyspace_name}/{self.astra_cassandra_configurer.database_name}/rows?page-size=1000"
            all_data = astra_http_client.request(method=http_methods.GET,path=url)
            #print(len(all_data['data']))
            rows = all_data['data']
            #print(rows)
            df = pd.DataFrame.from_dict(rows,orient="columns")
            return df

        except Exception as e:
            raise ToxicityException(e, sys)