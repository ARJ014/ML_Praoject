import sys
import os
import numpy as np
import pandas as pd
import pickle


from error import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(path, obj):
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        pickle.dump(obj, open(path, "wb"))

    except Exception as e:
        return CustomException(e, sys)


def load_object(path):
    try:
        return pickle.load(open(path, "rb"))

    except Exception as e:
        return CustomException(e, sys)


def evaluate_models(x_trian, x_test, y_train, y_test, models, params):
    report = {}
    try:
        for i in range(len(models)):
            model = list((models.values()))[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(
                estimator=model, param_grid=param, n_jobs=-1, verbose=1, cv=3
            )
            gs.fit(x_trian, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_trian, y_train)
            y_train_pred = model.predict(x_trian)
            y_test_pred = model.predict(x_test)

            train_accuaracy = r2_score(y_train, y_train_pred)
            test_accuaracy = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_accuaracy

        return report

    except Exception as e:
        # print(e)
        raise CustomException(e, sys)
