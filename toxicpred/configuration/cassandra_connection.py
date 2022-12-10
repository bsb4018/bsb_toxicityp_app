import os
import sys

from toxicpred.constant.database import KEYSPACE_NAME,DATABASE_NAME
from toxicpred.constant.env_variable import ASTRA_CLUSTER_ID, ASTRA_REGION, ASTRA_DB_APPLICATION_TOKEN
from toxicpred.exception import ToxicityException
from pathlib import Path
from astrapy.rest import create_client, http_methods


class AstraCassandraConfig:

    def __init__(self) -> None:
        try:
            self.keyspace_name = KEYSPACE_NAME
            self.database_name = DATABASE_NAME
            self.astra_cluster_id = os.getenv(ASTRA_CLUSTER_ID)
            self.astra_region = os.getenv(ASTRA_REGION)
            self.astra_db_application_token = os.getenv(ASTRA_DB_APPLICATION_TOKEN)

        except Exception as e:
            raise ToxicityException(e, sys)

    def getAstraHTTPClient(self):
        try:
             return create_client(astra_database_id=self.astra_cluster_id,
                         astra_database_region=self.astra_region,
                         astra_application_token=self.astra_db_application_token)

        except Exception as e:
            raise ToxicityException(e, sys)

        

