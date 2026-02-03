import sys
import os
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        tuned_models = {}

        for name, model in models.items():
            para = param.get(name, {})
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)

            tuned_model = gs.best_estimator_
            tuned_model.fit(X_train, y_train)

            y_train_pred = tuned_model.predict(X_train)
            y_test_pred = tuned_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[name] = test_model_score
            tuned_models[name] = tuned_model

        return report, tuned_models

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)