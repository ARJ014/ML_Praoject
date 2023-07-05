import sys
import os
import numpy as np
import pandas as pd
import pickle

from error import CustomException
from sklearn.metrics import r2_score


def save_object(path, obj):
    try:
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        pickle.dump(obj, open(path, "wb"))

    except Exception as e:
        return CustomException(e, sys)


def evaluate_models(x_trian, x_test, y_train, y_test, models):
    report = {}
    try:
        for i in range(len(models)):
            model = list((models.values()))[i]
            model.fit(x_trian, y_train)

            y_train_pred = model.predict(x_trian)
            y_test_pred = model.predict(x_test)

            train_accuaracy = r2_score(y_train, y_train_pred)
            test_accuaracy = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_accuaracy

        return report

    except Exception as e:
        return CustomException(e, sys)
