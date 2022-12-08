from pandas import DataFrame
from toxicpred.exception import ToxicityException
import sys
from toxicpred.cloud_storage.s3_syncer import S3Sync
from toxicpred.ml.model.estimator import ModelResolver, ToxicityModel
from toxicpred.constant.training_pipeline import SAVED_MODEL_DIR
from toxicpred.utils.main_utils import read_yaml_file, read_json_file
from toxicpred.constant.training_pipeline import SCHEMA_FILE_PATH, VALID_SCHEMA_FILE_PATH
from toxicpred.logger import logging
from toxicpred.utils.main_utils import load_object
import warnings
warnings.filterwarnings("ignore")

class ModelPrediction:
    def __init__(self):
        try:
            self.model_resolver_local = ModelResolver(model_dir=SAVED_MODEL_DIR)
            self.s3_sync = S3Sync()
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
            self._valid_schema = read_json_file(file_path=VALID_SCHEMA_FILE_PATH)
        except Exception as e:
            raise ToxicityException(e,sys)


    def check_data_validity(self, df: DataFrame) -> bool:
        try:
            logging.info("Entered the check_data_validity method of Model Prediction class")
            status = True

            #numerical columns
            cico_min = self._valid_schema["CIC0"]["min"]
            cico_max = self._valid_schema["CIC0"]["max"]
            cico_val_list = list(df["CIC0"].values)

            for val in cico_val_list:
                if val < cico_min or val > cico_max:
                    status = False
                    logging.info("Invalid data found!! Exiting the check_data_validity method of Model Prediction class")
                    return status

            gats_min = self._valid_schema["GATS1i"]["min"]
            gats_max = self._valid_schema["GATS1i"]["max"]
            gats_val_list = list(df["GATS1i"].values)

            for val in gats_val_list:
                if val < gats_min or val > gats_max:
                    status = False
                    logging.info("Invalid data found!! Exiting the check_data_validity method of Model Prediction class")
                    return status

            mlogp_min = self._valid_schema["MLOGP"]["min"]
            mlogp_max = self._valid_schema["MLOGP"]["max"]
            mlogp_val_list = list(df["MLOGP"].values)

            for val in mlogp_val_list:
                if val < mlogp_min or val > mlogp_max:
                    status = False
                    logging.info("Invalid data found!! Exiting the check_data_validity method of Model Prediction class")
                    return status

            smdz_min = self._valid_schema["SM1_DzZ"]["min"]
            smdz_max = self._valid_schema["SM1_DzZ"]["max"]
            smdz_val_list = list(df["SM1_DzZ"].values)
            
            for val in smdz_val_list:
                if val < smdz_min or val > smdz_max:
                    status = False
                    logging.info("Invalid data found!! Exiting the check_data_validity method of Model Prediction class")
                    return status

            #categorical columns
            valid_ndsch_list = self._valid_schema["NdsCH"]
            ndsch_values_list = list(df['NdsCH'].values)

            valid_ndssc_list = self._valid_schema["NdssC"]
            ndssc_values_list = list(df['NdsCH'].values)

            for val in ndsch_values_list:
                if val not in valid_ndsch_list:
                    status = False
                    logging.info("Invalid data found!! Exiting the check_data_validity method of Model Prediction class")
                    return status

            for val in ndssc_values_list:
                if val not in valid_ndssc_list:
                    status = False
                    logging.info("Invalid data found!! Exiting the check_data_validity method of Model Prediction class")
                    return status

            logging.info("All data valid, Exiting the check_data_validity method of Model Prediction class")            
            return status
   
        except Exception as e:
            raise ToxicityException(e, sys) from e

    
    def predict_output(self, df:DataFrame):
        try:
            logging.info("Entered the predict_output method of Model Prediction class")
            model_resolver = self.model_resolver_local
            if not model_resolver.is_model_exists():
                return []
            best_model_path = model_resolver.get_best_model_path()
            model = load_object(file_path=best_model_path)
            y_pred = model.predict(df)
            df['predicted_column'] = y_pred
            prediction_result = df['predicted_column'].tolist()
            logging.info("Exiting the predict_output method of Model Prediction class")
            return prediction_result

        except Exception as e:
            raise ToxicityException(e,sys) from e