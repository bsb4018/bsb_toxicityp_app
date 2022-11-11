from toxicpred.constant.training_pipeline import SCHEMA_FILE_PATH
from toxicpred.utils.main_utils import read_yaml_file
import pandas as pd

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
