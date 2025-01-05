import pandas as pd
from train import train
import os
from plots import makeplots
import pickle
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from project.reporting.generate_report import GenerateReport

class medaid:
    allowed_models = ["logistic", "tree", "random_forest", "xgboost", "lightgbm"]
    allowed_metrics = [ "accuracy", "f1", "recall", "precision"] #TODO ktore metryki ?
    def __init__(self
                 , X
                 , y
                 , mode = "perform"
                 , models = None
                 , metric = "f1"
                 , path = None
                 , search  = None
                 , cv = 3
                 , n_iter = 20
                 ):

        if type(X) is not pd.DataFrame:
            raise ValueError("X must be a pandas DataFrame")
        if type(y) is not pd.Series and type(y) is not pd.DataFrame:
            raise ValueError("y must be a pandas Series or DataFrame")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        self.X = X
        self.y = y

        if mode not in ["explain", "perform"]:
            raise ValueError("mode must be either 'explain' or 'perform'")
        self.mode = mode

        if models is not None:
            if type(models) is not list:
                raise ValueError("models must be a list")
            for model in models:
                if model not in self.allowed_models:
                    raise ValueError(f"model {model} is not allowed, must be one of {self.allowed_models}")
            self.models = models
        elif mode == "perform":
            self.models = self.allowed_models
        else:
            self.models = ["logistic", "tree"]


        if metric not in self.allowed_metrics:
            raise ValueError(f"metric {metric} is not allowed, must be one of {self.allowed_metrics}")
        self.metric = metric

        self.best_models_scores = None
        self.best_models = None
        self.best_metrics = None

        if path:
            self.path = path + "/medaid"
        else:
            self.path = os.path.dirname(os.path.abspath(__file__)) + "/medaid"

        if search:
            if search not in ["random", "grid"]:
                raise ValueError("search must be either 'random' or 'grid'")
            self.search = search
        else:
            self.search = "random" if mode == "explain" else "grid"

        if type(cv) is not int:
            raise ValueError("cv must be an integer")
        self.cv = cv
        if type(n_iter) is not int:
            raise ValueError("n_iter must be an integer")
        self.n_iter = n_iter

    def __repr__(self):
        # TODO: trzeba jakos ladnie zaprezentowac i guess - techniczna wizualizacja
        return f"medaid(X, y, mode = {self.mode})"

    def __str__(self):
        # TODO trzeba jakos ladnie zaprezentowac i guess - ladna wizualizacja
        str = "medaid object\n"
        str+=f"mode: {self.mode}\n"
        str+=f"metric: {self.metric}\n"
        if self.best_models is not None:
            str+="trained\n"
            str+="models; scores: \n"
            for i in range(len(self.best_models)):
                str+=f"\t- {self.best_models[i]}: {self.best_models_scores[i]}\n"
        else:
            str+="not trained\n"

        return str

    def train(self):
        best_models, best_models_scores, best_metrics= train(self.X, self.y, self.models, self.metric, self.mode, self.path, self.search, self.cv, self.n_iter)
        self.best_models = best_models
        self.best_models_scores = best_models_scores
        self.best_metrics = best_metrics
        makeplots(self.best_models, self.X, self.y, self.path)

    def predict(self, X):
        if self.best_models is None:
            raise ValueError("You need to train the model first")
        if type(X) is not pd.DataFrame:
            raise ValueError("X must be a pandas DataFrame")
        if len(X.columns) != len(self.X.columns):
            raise ValueError("X must have the same columns as the training data")
        return self.best_models[0].predict(X)

    def models_ranking(self):
        return self.best_metrics

    def report(self):
        pass

    def save(self):
        with open(f"{self.path}/medaid.pkl", 'wb') as f:
            pickle.dump(self, f)

