import os

SAVED_MODEL_DIR = os.path.join("saved_models")
TARGET_COLUMN = "responseLC50"
PIPELINE_NAME: str = "toxicity"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "toxicitypred.csv"


'''
Defining basic and common file names
'''
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"


'''
Data Ingestion related constant start with DATA_INGESTION VAR NAME
'''
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2
DATA_INGESTION_RANDOM_STATE: int = 42

 #DATA_INGESTION_RANDOM_STATE: int = 34         'logs\11_12_2022_03_12_12.log' 'artifact\11_12_2022_03_12_12'
 #DATA_INGESTION_RANDOM_STATE: int = 38 #best   'logs\11_12_2022_03_14_44.log' 'artifact\11_12_2022_03_14_44'


'''
Data Validation related constant start with DATA_VALIDATION VAR NAME
'''
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_DASHBOARD_DIR: str = "drift_report"
DATA_VALIDATION_DASHBOARD_FILE_NAME: str = "dashboard_report.html"

'''
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
'''
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


'''
MODEL TRAINER related constant start with MODEL_TRAINER var name
'''
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.5
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.20


'''
MODEL EVALUATION ralated constant start with MODEL_EVALUATION VAR NAME
'''

MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.005
MODEL_EVALUATION_REPORT_NAME= "report.yaml"


'''
Model Pusher ralated constant start with MODEL_PUSHER VAR NAME
'''
MODEL_PUSHER_DIR_NAME = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODEL_DIR