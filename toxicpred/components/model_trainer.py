import os,sys

from toxicpred.exception import ToxicityException
from toxicpred.logger import logging
from toxicpred.utils.main_utils import (load_numpy_array_data, load_object,
                                     save_object)
from toxicpred.entity.artifact_entity import (DataTransformationArtifact,
                                           ModelTrainerArtifact)
from toxicpred.entity.config_entity import ModelTrainerConfig
from toxicpred.ml.metric.regression_metric import get_regression_score
from toxicpred.ml.model.estimator import ToxicityModel
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

class ModelTrainer:
    def __init__(self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise ToxicityException(e,sys) 

    def train_model(self, x_train, y_train):
        try:
            model_svr = SVR(C = 0.9808155406244474, coef0 = 1.0617585221865262,\
                           degree = 3, epsilon = 0.1797523576454031, gamma = 'scale', 
                           kernel = 'rbf', max_iter = 300, tol = 0.027295414375092093)
            model_svr.fit(x_train, y_train)
            return model_svr
        except Exception as e:
            raise e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:

        try:
            logging.info("Entered initiate_model_trainer method of ModelTrainer class")

            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            #Load transformed data
            train_arr = load_numpy_array_data(
                file_path = train_file_path
            )
            test_arr = load_numpy_array_data(
                file_path = test_file_path
            )
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            model = self.train_model(x_train, y_train)
            y_train_pred = model.predict(x_train)
            regression_train_metric = get_regression_score(y_true = y_train, y_pred = y_train_pred)

            if regression_train_metric.r2_score<=self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good to provide expected accuracy")

            y_test_pred = model.predict(x_test)
            regression_test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)

            #Overfitting and Underfitting
            diff = abs(regression_train_metric.r2_score-regression_test_metric.r2_score)
            
            if diff>self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good try to do more experimentation.")

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            toxicity_model = ToxicityModel(preprocessor=preprocessor,model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=toxicity_model)

            #Model Trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
                train_metric_artifact=regression_train_metric,
                test_metric_artifact=regression_test_metric)
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise ToxicityException(e,sys) from e
