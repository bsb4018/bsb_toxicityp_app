from toxicpred.exception import ToxicityException
from toxicpred.logger import logging
from toxicpred.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from toxicpred.entity.config_entity import ModelEvaluationConfig
import os,sys
from toxicpred.ml.metric.regression_metric import get_regression_score
from toxicpred.ml.model.estimator import ToxicityModel
from toxicpred.utils.main_utils import save_object,load_object,write_yaml_file,write_json_file,read_yaml_file
from toxicpred.constant.training_pipeline import SCHEMA_FILE_PATH
from toxicpred.ml.model.estimator import ModelResolver
from toxicpred.constant.training_pipeline import TARGET_COLUMN
import pandas  as  pd
import warnings
warnings.filterwarnings("ignore")
class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                       data_validation_artifact: DataValidationArtifact,
                       data_transformation_artifact: DataTransformationArtifact,
                       model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise ToxicityException(e,sys) from e


    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path
            
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            df = pd.concat([train_df,test_df])
            y_true = df[TARGET_COLUMN]
            df.drop(TARGET_COLUMN, axis=1, inplace=True)

            numerical_columns = self._schema_config["numerical_columns"]
            data_transform_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
    
            transformed_npy_array = data_transform_obj.transform(df[numerical_columns])
            transformed_df = pd.DataFrame(transformed_npy_array, columns=numerical_columns)
            transformed_df.index=df.index
            categorical_columns = self._schema_config["categorical_columns"]
            df_final = pd.concat([transformed_df,df[categorical_columns]],axis=1)

            print(df_final.head(5))
     

            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            train_model = load_object(file_path=train_model_file_path)
            y_trained_pred = train_model.predict(df_final)
            trained_metric = get_regression_score(y_true, y_trained_pred)

            print("All Data Metrics")
            print("All_data_R2 score :",trained_metric.r2_score)
            print("All_data_RMSE :", trained_metric.mae_value)
            print("All_data_MAE :", trained_metric.rmse_value)


            model_resolver = ModelResolver()
            is_model_accepted = True

            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_path=train_model_file_path,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.train_metric_artifact,
                    best_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact
                )

                model_eval_report = {"r2_score" : trained_metric.r2_score, "mae_value": trained_metric.mae_value, "rmse_value": trained_metric.rmse_value}
                write_json_file(self.model_eval_config.report_file_path, model_eval_report)
                
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            y_latest_pred = latest_model.predict(df_final)
            latest_metric = get_regression_score(y_true, y_latest_pred)
            

            improved_accuracy = trained_metric.r2_score-latest_metric.r2_score
            if self.model_eval_config.change_threshold < improved_accuracy:
                is_model_accepted=True
            else:
                is_model_accepted=False

            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=improved_accuracy, 
                    best_model_path=latest_model_path, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=trained_metric, 
                    best_model_metric_artifact=latest_metric)
              
            model_eval_report = {"r2_score" : trained_metric.r2_score, "mae_value": trained_metric.mae_value, "rmse_value": trained_metric.rmse_value}
            write_json_file(self.model_eval_config.report_file_path, model_eval_report)


            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise ToxicityException(e,sys) from e