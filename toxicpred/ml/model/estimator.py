from toxicpred.constant.training_pipeline import SCHEMA_FILE_PATH
from toxicpred.utils.main_utils import read_yaml_file
import pandas as pd
import os
from toxicpred.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME


class ToxicityModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise e
    
    def predict(self,x: pd.DataFrame):
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            categorical_columns = self._schema_config["categorical_columns"]
            x_transform = self.preprocessor.transform(x[numerical_columns])
            x_transform_df = pd.DataFrame(x_transform, columns = numerical_columns)
            x_transform_df_main = pd.concat([x_transform_df,x[categorical_columns]],axis=1)
            y_hat = self.model.predict(x_transform_df_main)
            return y_hat
        except Exception as e:
            raise e


class ModelResolver:
    def __init__(self, model_dir=SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir

        except Exception as e:
            raise e

    def get_best_model_path(self,) -> str:
        try:
            timestamps = list(map(int, os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)
            latest_model_path = os.path.join(self.model_dir, f"{latest_timestamp}", MODEL_FILE_NAME)
            return latest_model_path       
        
        except Exception as e:
            raise e

    def is_model_exists(self) -> bool:
        try:
            if not os.path.exists(self.model_dir):
                return False

            timestamps = os.listdir(self.model_dir)
            if len(timestamps) == 0:
                return False

            latest_model_path = self.get_best_model_path()
            if not os.path.exists(latest_model_path):
                return False

            return True

        except Exception as e:
            raise e