import os
import sys

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from toxicpred.data_access.toxicity_data import ToxicityData
from toxicpred.entity.artifact_entity import DataIngestionArtifact
from toxicpred.entity.config_entity import DataIngestionConfig
from toxicpred.exception import ToxicityException
from toxicpred.logger import logging
from toxicpred.utils.main_utils import read_yaml_file
from toxicpred.constant.training_pipeline import SCHEMA_FILE_PATH


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise ToxicityException(e,sys)

    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Exporting data from astra cassandra database")
            toxic_data = ToxicityData()
            dataframe = toxic_data.export_from_astra_database_to_dataframe()
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(
                f"Saving exported data into feature store file path: {feature_store_file_path}"
            )
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe

        except Exception as e:
            raise ToxicityException(e, sys)
    
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        
        try:
            logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path.")
        
        except Exception as e:
            raise ToxicityException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:

        try:
            logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

            dataframe = self.export_data_into_feature_store()
            dataframe = dataframe.drop(self._schema_config["drop_columns"], axis=1)
            dataframe.replace({"na": np.nan}, inplace=True)
            logging.info("Got the data from mongodb")
            self.split_data_as_train_test(dataframe)
            logging.info("Performed train test split on the dataset")
            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise ToxicityException(e, sys) from e
