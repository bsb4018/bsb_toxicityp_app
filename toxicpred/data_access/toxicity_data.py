import sys
import numpy as np
import pandas as pd
from toxicpred.exception import ToxicityException
from toxicpred.configuration.cassandra_connection import AstraCassandraConfig
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

class ToxicityData:
    """
    Export data from astra cassandra database to pandas dataframe
    """
    def __init__(self):
        try:
            self.astra_cassandra_configurer = AstraCassandraConfig()
        except Exception as e:
            raise ToxicityException(e, sys)

    def export_from_astra_database_to_dataframe(self) -> pd.DataFrame:
        try:
            cloud_config = {
                'secure_connect_bundle': self.astra_cassandra_configurer.secure_connect_bundle_path
            }
            auth_provider = PlainTextAuthProvider(self.astra_cassandra_configurer.client_id, self.astra_cassandra_configurer.client_secret)
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider, protocol_version=4)
            session = cluster.connect(self.astra_cassandra_configurer.keyspace_name)
            rows = session.execute(f"select * from {self.astra_cassandra_configurer.database_name}")
            
            df = pd.DataFrame(rows)
            return df

        except Exception as e:
            raise ToxicityException(e,sys)