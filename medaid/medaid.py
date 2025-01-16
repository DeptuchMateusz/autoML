import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from medaid.preprocessing.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from medaid.training.train import train
from medaid.reporting.plots import makeplots
from medaid.reporting.mainreporter import MainReporter
from medaid.reporting.predictexplain import PredictExplainer
import pickle
import sys
import os
import numpy as np

class MedAId:
    allowed_models = ["logistic", "tree", "random_forest", "xgboost", "lightgbm"]
    allowed_metrics = [ "accuracy", "f1", "recall", "precision"] #TODO ktore metryki ?
    def __init__(self
                 , dataset_path
                 , target_column
                 , models=None
                 , metric = "f1"
                 , path = None
                 , search  = 'random'
                 , cv = 3
                 , n_iter = 30
                 , test_size = 0.2
                 , n_jobs = -1
                 , param_grids = None
                 , imputer_lr_correlation_threshold=0.8
                 , imputer_rf_correlation_threshold=0.2
                 , removal_threshold=0.2
                 , removal_correlation_threshold=0.9
                 ):

        self.dataset_path = dataset_path
        self.target_column = target_column


        if models is not None:
            if type(models) is not list:
                raise ValueError("models must be a list")
            for model in models:
                if model not in self.allowed_models:
                    raise ValueError(f"model {model} is not allowed, must be one of {self.allowed_models}")
            self.models = models
        else:
            self.models = self.allowed_models



        if metric not in self.allowed_metrics:
            raise ValueError(f"metric {metric} is not allowed, must be one of {self.allowed_metrics}")
        self.metric = metric

        self.best_models_scores = None
        self.best_models = None
        self.best_metrics = None


        if path:
            self.path = path + "/medaid1"
        else:
            self.path = os.getcwd() + "/medaid1"


        counter = 1
        original_path = os.getcwd() + "/medaid"
        while os.path.exists(self.path):
            self.path = f"{original_path}{counter}"
            counter += 1

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + "/results", exist_ok=True)
        os.makedirs(self.path + "/results/models", exist_ok=True)



        if search not in ["random", "grid"]:
            raise ValueError("search must be either 'random' or 'grid'")
        self.search = search

        self.imputer_lr_correlation_threshold = imputer_lr_correlation_threshold
        self.imputer_rf_correlation_threshold = imputer_rf_correlation_threshold
        self.removal_threshold = removal_threshold
        self.removal_correlation_threshold = removal_correlation_threshold        
        self.preprocess = Preprocessing(self.target_column, self.path, self.imputer_lr_correlation_threshold,
                                        self.imputer_rf_correlation_threshold, self.removal_threshold,
                                        self.removal_correlation_threshold)

        if type(cv) is not int:
            raise ValueError("cv must be an integer")
        self.cv = cv
        if type(n_iter) is not int:
            raise ValueError("n_iter must be an integer")
        self.n_iter = n_iter

        self.test_size = test_size

        self.df_before = self.read_data()
        self.df = self.read_data()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_jobs = n_jobs

        if param_grids:
            self.param_grids = param_grids
        else:
            self.param_grids = {
                "logistic": {
                    'C': list(np.logspace(-3, 3, 20)),  # 20 values from 1e-3 to 1e3
                    'penalty': ['l2'],
                    'solver': [
                        'saga',
                        'lbfgs',
                        'newton-cg'
                    ]
                },
                "tree": {
                    'max_depth': [3, 4, 5, 7, 9, 11, 13, 15],
                    'min_samples_split': [2, 4, 6, 8, 10],
                    'min_samples_leaf': [1, 2, 3, 4, 5]
                },
                "random_forest": {
                    'n_estimators': [50, 100, 200, 300, 400, 500],
                    'max_depth': [None, 3, 5, 8, 11, 15, 20],
                    'min_samples_split': [2, 3, 4, 6, 8, 10],
                    'min_samples_leaf': [1, 2, 3, 5, 7, 10],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False],
                },

                "xgboost": {
                    'verbosity': [0],
                    'n_estimators': [50, 100, 200, 300, 400, 500],
                    'max_depth': [3, 5, 7, 9, 11, 13, 15],
                    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
                    'subsample': [0.5, 0.7, 0.8, 1],
                    'colsample_bytree': [0.5, 0.75, 1],
                    'colsample_bylevel': [0.5, 0.75, 1],
                    'reg_alpha': [0, 0.01, 0.1, 0.5, 1, 10],
                    'reg_lambda': [0, 0.01, 0.1, 0.5, 1, 10],
                    'gamma': [0, 0.01, 0.1, 0.5, 1],
                    'min_child_weight': [1, 3, 5, 7, 10],
                    'scale_pos_weight': [1, 10, 50, 100],
                    'tree_method': ['auto', 'exact', 'approx', 'hist']
                },

                "lightgbm": {
                    'verbosity': [-1],
                    'learning_rate': [0.005, 0.01, 0.05, 0.1],
                    'n_estimators': [50, 100, 200, 300],
                    'num_leaves': [6, 8, 12, 16, 24, 32],
                    'boosting_type': ['gbdt', 'dart', 'goss'],
                    'max_bin': [255, 510, 1023],
                    'random_state': [500],
                    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
                    'subsample': [0.7, 0.75, 0.8, 0.9],
                    'reg_alpha': [0, 0.5, 1, 1.2, 2, 5],
                    'reg_lambda': [0, 0.5, 1, 1.2, 1.4, 2, 5],
                }
            }


    def __repr__(self):
        return f"medaid(X, y, models={self.models}, metric={self.metric}, path={self.path}, search={self.search}, cv={self.cv}, n_iter={self.n_iter})"

    def __str__(self):
        str = "medaid object\n"
        str+=f"metric: {self.metric}\n"
        if self.best_models is not None:
            str+="trained\n"
            str+="models; scores: \n"
            for i in range(len(self.best_models)):
                str+=f"\t- {self.best_models[i]}: {self.best_models_scores[i]}\n"
        else:
            str+="not trained\n"

        return str

    def read_data(self):
        if not os.path.exists(self.dataset_path):
            raise ValueError(f"File not found at {self.dataset_path}. Please make sure the file exists.")
        if self.dataset_path.endswith(".csv"):
            return pd.read_csv(self.dataset_path, sep=None, engine='python')
        if self.dataset_path.endswith(".xlsx") or self.dataset_path.endswith(".xls"):
            return pd.read_excel(self.dataset_path)
        else:
            raise ValueError(f"File format not supported. Please provide a CSV or Excel file.")

    def preprocessing(self):
        return self.preprocess.preprocess(self.df_before)


    def split_and_validate_data(self, test_size=0.2, max_attempts=50):
        all_classes = set(self.y)  

        for attempt in range(max_attempts):
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, stratify=self.y, random_state=42 + attempt
            )
            
            train_classes = set(y_train)

            if all_classes.issubset(train_classes):
                return X_train, X_test, y_train, y_test

        raise ValueError("Could not ensure all classes are present in the training set after maximum attempts.")


    def train(self):
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        df = self.preprocessing()
        self.X = df.drop(columns=[self.target_column])
        self.y = df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_and_validate_data(test_size=self.test_size)

        best_models, best_models_scores, best_metrics= train(self.X_train, self.y_train,self.X_test, self.y_test,
                                                             self.models, self.metric, self.path, self.search,
                                                             self.cv, self.n_iter, self.n_jobs, self.param_grids)
        self.best_models = best_models
        self.best_models_scores = best_models_scores
        self.best_metrics = best_metrics

        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        makeplots(self)

        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    def predict(self, X):
        if self.best_models is None:
            raise ValueError("You need to train the model first")
        if type(X) is not pd.DataFrame or type(X) is not pd.Series:
            raise ValueError("X must be a pandas DataFrame")
        if len(X.columns) != len(self.X.columns):
            raise ValueError("X must have the same columns as the training data")
        prediction = self.best_models[0].predict(X)
        #TODO decode

        return prediction

    def models_ranking(self):
        return self.best_metrics

    def report(self):
        MainReporter(self, self.path).generate_report()

    def save(self):
        with open(f"{self.path}/medaid.pkl", 'wb') as f:
            pickle.dump(self, f)

    def predict_explain(self, input_data = None, model= None):
        warnings.filterwarnings("ignore", category=FutureWarning)
        if not model:
            model = self.best_models[0]
        if not input_data:
            input_data = self.df_before.head(1).drop(columns=[self.target_column])
        pe = PredictExplainer(self, model)
        df = self.df_before.drop(columns=[self.target_column])
        html_report = pe.generate_html_report(df, input_data)
        with open(f"{self.path}/prediction_report.html", 'w') as f:
            f.write(html_report)
