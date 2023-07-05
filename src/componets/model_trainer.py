from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from logger import logging
from error import CustomException
from dataclasses import dataclass

import utils
import os
import sys


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            x_train, x_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1],
            )
            print(x_train.shape, y_train.shape)
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: dict = utils.evaluate_models(
                x_train, x_test, y_train, y_test, models
            )

            for key, value in model_report.items():
                print(f"Model:{key}, score={value}")

            best_model_score = max(sorted(list(model_report.values())))

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            logging.info("Best model has been found")

            utils.save_object(
                path=self.model_trainer_config.trained_model_file_path, obj=best_model
            )

            r2 = r2_score(y_test, best_model.predict(x_test))
            return r2

        except Exception as e:
            return CustomException(e, sys)
