import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets for training, validation and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    X[X.select_dtypes('object').columns] = X[X.select_dtypes('object').columns].astype("category")
    y = data["target"].astype("category")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=(parameters["val_size"]/parameters["train_size"]), random_state=parameters["random_state"]
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict):
    """Trains the XGBoost model.

    Args:
        X_train: Training data of client features.
        y_train: Training data for target product (DI).
        parameters: Parameters defined in parameters/data_science.yml.

    Returns:
        Trained model.
    """
    dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
    param = {'max_depth': parameters['max_depth'],
             'eta': parameters['eta'],
             'gamma': parameters['gamma'],
             'min_child_weight': parameters['min_child_weight'],
             'colsample_bytree': parameters['colsample_bytree'],
             'objective': parameters['objective'],
             'eval_metric': parameters['eval_metric']}
    booster = xgb.train(dtrain=dtrain, params=param, num_boost_round=parameters['nrounds'])
    return booster


def evaluate_model(booster, X_test: pd.DataFrame, y_test: pd.Series, parameters: Dict):
    """Calculates and logs the ROC AUC score.

    Args:
        booster: Trained model.
        X_test: Testing data of client features.
        y_test: Testing data for target variable.
        parameters: Parameters defined in parameters/data_science.yml.
    """
    dtest = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True)
    y_pred = booster.predict(dtest)
    score = roc_auc_score(y_test, y_pred)
    print(f"Model {parameters['model_name']} has an ROC AUC of {score} on the test data.")
    logger = logging.getLogger(__name__)
    logger.info(f"Model {parameters['model_name']} has an ROC AUC of {score} on the test data.")


# def tune_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, parameters: Dict):
#     """Tunes the hyperparameters of the XGBoost model.
#
#         Args:
#             X_train: Training data of features.
#             y_train: Training data for target variable.
#             X_val: Validation data of features.
#             y_val: Validation data for target variable.
#             parameters: Parameters defined in parameters/data_science.yml.
#
#         Returns:
#             Dict of best performing model hyperparameters.
#         """
#     pass