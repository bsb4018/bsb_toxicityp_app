import os,sys
import numpy as np
from toxicpred.exception import ToxicityException
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from toxicpred.entity.artifact_entity import RegressionMetricArtifact

def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    try:
       
        model_rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
        model_r2_score = r2_score(y_true, y_pred)
        model_mae_value = mean_absolute_error(y_true,y_pred)

        regression_metric =  RegressionMetricArtifact(
                    rmse_value = model_rmse_value,
                    r2_score = model_r2_score, 
                    mae_value = model_mae_value)
        return regression_metric
    except Exception as e:
        raise ToxicityException(e,sys)
