import sys
from toxicpred.exception import ToxicityException
from toxicpred.logger import logging
from pandas import DataFrame
from toxicpred.ml.model.estimator import ModelResolver, ToxicityModel
from toxicpred.constant.training_pipeline import SAVED_MODEL_DIR
from toxicpred.utils.main_utils import load_object
from toxicpred.cloud_storage.s3_syncer import S3Sync

class PredictionPipeline:
    def __init__(self):
        #self.toxicity_model = ToxicityModel(preprocessor=preprocessor,model=model)
        self.model_resolver_local = ModelResolver(model_dir=SAVED_MODEL_DIR)
        self.s3_sync = S3Sync()
    
    def predict(self, df:DataFrame):
        try:
            logging.info("Entered the predict_from_local method of PredictionPipeline class")
            model_resolver = self.model_resolver_local
            if not model_resolver.is_model_exists():
                return []
            best_model_path = model_resolver.get_best_model_path()
            model = load_object(file_path=best_model_path)
            y_pred = model.predict(df)
            df['predicted_column'] = y_pred
            prediction_result = df['predicted_column'].tolist()
            return prediction_result

        except Exception as e:
            raise ToxicityException(e,sys) from e
    
        

