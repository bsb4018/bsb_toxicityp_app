import sys
from toxicpred.exception import ToxicityException
from toxicpred.logger import logging
from pandas import DataFrame
from toxicpred.components.model_prediction import ModelPrediction
class PredictionPipeline:
    def __init__(self):
        try:
            self.prediction_component = ModelPrediction()
        except Exception as e:
            raise ToxicityException(e,sys) from e

    def validate(self, df:DataFrame) -> bool:
        try:
            logging.info("Entered the validate method of PredictionPipeline class")
            status = self.prediction_component.check_data_validity(df)
            logging.info("Exiting the validate method of PredictionPipeline class")
            return status
        
        except Exception as e:
            raise ToxicityException(e,sys) from e
    
    def predict(self, df:DataFrame):
        try:
            logging.info("Entered the predict method of PredictionPipeline class")
            prediction_result = self.prediction_component.predict_output(df)
            logging.info("Exiting the predict method of PredictionPipeline class")
            return prediction_result

        except Exception as e:
            raise ToxicityException(e,sys) from e
    
        

