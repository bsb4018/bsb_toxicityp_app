import os, sys
import json
import pandas as pd
from scipy.stats import ks_2samp
from pandas import DataFrame
from toxicpred.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from toxicpred.entity.config_entity import DataValidationConfig
from toxicpred.constant.training_pipeline import SCHEMA_FILE_PATH
from toxicpred.exception import ToxicityException
from toxicpred.logger import logging
from toxicpred.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise ToxicityException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ToxicityException(e, sys) from e

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        '''
        Takes input a dataframe and returns 'True' 
        if all required columns are present
        '''
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"]) - 1
            logging.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            raise ToxicityException(e, sys) from e

    def is_numerical_column_exist(self, df: DataFrame) -> bool:
        '''
        Takes input a dataframe and returns 'True' if 
        all the designated numerical columns are present
        '''
        try:
            dataframe_columns = df.columns
            status = True
            missing_numerical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    status = False
                    missing_numerical_columns.append(column)
            
            logging.info(f"Missing numerical column: {missing_numerical_columns}")
            return status

        except Exception as e:
            raise ToxicityException(e, sys) from e

    def is_categorical_column_exist(self, df: DataFrame) -> bool:
        '''
        Takes input a dataframe and returns 'True' if 
        all the designated categorical columns are present
        '''
        try:
            dataframe_columns = df.columns
            status = True
            missing_categorical_columns = []
            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    status = False
                    missing_categorical_columns.append(column)
            
            logging.info(f"Missing categorical column: {missing_categorical_columns}")
            return status

        except Exception as e:
            raise ToxicityException(e, sys) from e

    def detect_dataset_drift(
        self, base_df: DataFrame, current_df: DataFrame, threshold=0.05
    ) -> bool:
        '''
        takes input two dataframes and returns 'True' or 'False'
        if there is dataset drift found
        '''
        try:
            status=True
            report ={}
            for column in base_df.columns:
                d1 = base_df[column]
                d2  = current_df[column]
                is_same_dist = ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found = True 
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            
            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report,)
            return status
        except Exception as e:
            raise ToxicityException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        '''
        initiates the complete data validation component
        '''
        try:
            logging.info("Starting data validation")

            train_df, test_df = (
                DataValidation.read_data(
                    file_path = self.data_ingestion_artifact.trained_file_path
                ),
                DataValidation.read_data(
                    file_path = self.data_ingestion_artifact.test_file_path
                )
            )

            #Validating number of columns
            validation_error_msg = ""
            validation_status = True

            status = self.validate_number_of_columns(dataframe=train_df)
            
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No columns are missing in training dataframe "
            
            status = self.validate_number_of_columns(dataframe=test_df)
            
            if not status:
                validation_error_msg += f"Columns are missing in testing dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No columns are missing in testing dataframe "

            logging.info(f"All Columns Validation Message: {validation_error_msg}")
            if not validation_status:
                raise Exception(validation_error_msg)
            
            #Validating numerical columns
            validation_error_msg = ""
            validation_status = True

            status = self.is_numerical_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Numerical columns are missing in training dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No numerical columns are missing in training dataframe "
                

            status = self.is_numerical_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Numerical columns are missing in testing dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No numerical columns are missing in testing dataframe "
                
            
            logging.info(f"Numerical Columns Validation Message: {validation_error_msg}")
            if not validation_status:
                raise Exception(validation_error_msg)

            
            #Validating categorical columns
            validation_error_msg = ""
            validation_status = True

            status = self.is_categorical_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Categorical columns are missing in training dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No categorical columns are missing in training dataframe "

            status = self.is_categorical_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Categorical columns are missing in testing dataframe "
                validation_status = False
            else:
                validation_error_msg += f"No categorical columns are missing in testing dataframe "

            logging.info(f"Categorical Columns Validation Message: {validation_error_msg}")
            if not validation_status:
                raise Exception(validation_error_msg)


            #Check Data Drift
            validation_error_msg = ""
            #validation_status = True
            status = self.detect_dataset_drift(base_df=train_df,current_df=test_df)
            if not status:
                validation_error_msg += f"Data Drift Detected "
                #validation_status = False
            else:
                validation_error_msg += f"No Data Drift Detected "
            logging.info(f"Data Drift Message: {validation_error_msg}")
            
            #STOP AND RAISE EXCEPTON FOR DATADRIFT IF REQUIRED HERE

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
  
            return data_validation_artifact
            
        except Exception as e:
            raise ToxicityException(e,sys) from e






  