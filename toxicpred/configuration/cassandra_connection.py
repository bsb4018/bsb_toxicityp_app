import os
import sys

from toxicpred.constant.database import KEYSPACE_NAME,DATABASE_NAME
from toxicpred.constant.env_variable import SECURE_CONNECT_BUNDLE_PATH, CLIENT_ID, CLIENT_SECRET
from toxicpred.exception import ToxicityException
from pathlib import Path

class AstraCassandraConfig:

    def __init__(self) -> None:
        try:
            self.keyspace_name = KEYSPACE_NAME
            self.database_name = DATABASE_NAME
            self.secure_connect_bundle_path = os.getenv(SECURE_CONNECT_BUNDLE_PATH)
            self.client_id = os.getenv(CLIENT_ID)
            self.client_secret = os.getenv(CLIENT_SECRET)

        except Exception as e:
            raise ToxicityException(e, sys)

