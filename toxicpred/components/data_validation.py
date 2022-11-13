import os, sys
import json
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import DataDriftTab
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
        self, base_df: DataFrame, current_df: DataFrame) -> bool:
        '''
        takes input two dataframes and returns 'True' or 'False'
        if there is dataset drift found
        '''
        try:

            data_drift_profile = Profile(sections=[DataDriftProfileSection()])

            data_drift_profile.calculate(base_df, current_df)

            report = data_drift_profile.json()
            json_report = json.loads(report)

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=json_report,
            )

            n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
            n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]


            data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
            data_drift_dashboard.calculate(base_df, current_df)
            #data_drift_dashboard.show()
            data_drift_dashboard.save(self.data_validation_config.drift_report_dashboard_path)

            logging.info(f"Drift detected in {n_drifted_features} out of {n_features}")
            drift_status = json_report["data_drift"]["data"]["metrics"]["dataset_drift"]
            return drift_status

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
                drift_report_dashboard_path= self.data_validation_config.drift_report_dashboard_path
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
  
            return data_validation_artifact
            
        except Exception as e:
            raise ToxicityException(e,sys) from e






  