import pandas as pd
from train import train


class medaid:
    allowed_models = ["logistic", "tree", "random_forest", "xgboost", "lightgbm"]
    allowed_metrics = [ "accuracy", "f1"] #TODO ktore metryki ?
    def __init__(self
                 , X
                 , y
                 , mode = "perform"
                 , models = None
                 , metric = "f1"

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


    def __repr__(self):
        # TODO: trzeba jakos ladnie zaprezentowac i guess - techniczna wizualizacja
        return f"medaid(X, y, mode = {self.mode})"

    def __str__(self):
        # TODO trzeba jakos ladnie zaprezentowac i guess - ladna wizualizacja
        return f"medaid(X, y, mode = {self.mode})"

    def train(self):
        best_models, best_models_scores = train(self.X, self.y, self.models, self.metric, self.mode)
        self.best_models = best_models
        self.best_models_scores = best_models_scores
