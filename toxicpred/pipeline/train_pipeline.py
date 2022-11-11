import sys
from toxicpred.components.data_ingestion import DataIngestion
from toxicpred.components.data_validation import DataValidation
from toxicpred.entity.config_entity import DataValidationConfig, TrainingPipelineConfig,DataIngestionConfig
from toxicpred.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from toxicpred.exception import ToxicityException
from toxicpred.logger import logging

class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info(
              "Entered the start_data_ingestion method of TrainPipeline class"
            )
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )
            return data_ingestion_artifact
    
        except Exception as e:
            raise ToxicityException(e, sys) from e
    
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
      
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=data_validation_config
            )

            data_validation_artifact = data_validation.initiate_data_validation()

            logging.info("Performed the data validation operation")
            logging.info(
                "Exited the start_data_validation method of TrainPipeline class"
            )

            return data_validation_artifact

        except Exception as e:
            raise ToxicityException(e,sys) from e


    
    def run_pipeline(self,) -> None:
        try:
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            TrainPipeline.is_pipeline_running=True
            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            

            logging.info("Training Pipeline Running Operation Complete")
            logging.info(
                "Exited the run_pipeline method of TrainPipeline class"
            )
        except Exception as e:
            TrainPipeline.is_pipeline_running=False
            raise ToxicityException(e, sys) from e