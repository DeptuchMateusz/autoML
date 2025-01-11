import pandas as pd
from project.preprocessing.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split

from project.do_poprawy_code.train import train
import os
from project.do_poprawy_code.plots import makeplots
import pickle
import sys
import os

class medaid:
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
                 , n_iter = 20
                 , test_size = 0.2
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
            self.path = path + "/medaid"
        else:
            self.path = os.getcwd() + "/medaid"


        counter = 1
        original_path = self.path
        while os.path.exists(self.path):
            self.path = f"{original_path}{counter}"
            counter += 1

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + "/results", exist_ok=True)
        os.makedirs(self.path + "/results/models", exist_ok=True)



        if search not in ["random", "grid"]:
            raise ValueError("search must be either 'random' or 'grid'")
        self.search = search

        self.preprocess = Preprocessing(target_column, self.path)

        if type(cv) is not int:
            raise ValueError("cv must be an integer")
        self.cv = cv
        if type(n_iter) is not int:
            raise ValueError("n_iter must be an integer")
        self.n_iter = n_iter

        self.test_size = test_size

        self.df = self.read_data()
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def __repr__(self):
        # TODO: trzeba jakos ladnie zaprezentowac i guess - techniczna wizualizacja
        return f"medaid(X, y, models={self.models}, metric={self.metric}, path={self.path}, search={self.search}, cv={self.cv}, n_iter={self.n_iter})"

    def __str__(self):
        # TODO trzeba jakos ladnie zaprezentowac i guess - ladna wizualizacja
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
        if self.dataset_path.endswith(".xlsx"):
            return pd.read_excel(self.dataset_path)

    def preprocessing(self, df):
        return self.preprocess.preprocess(df)


    def train(self):
        df = self.preprocessing(self.df)
        self.X = df.drop(columns=[self.target_column])
        self.y = df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=42)

        best_models, best_models_scores, best_metrics= train(self.X_train, self.y_train,self.X_test, self.y_test, self.models, self.metric, self.path, self.search, self.cv, self.n_iter)
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
        # Ensure log.txt is created
        if not os.path.exists('log.txt'):
            open('log.txt', 'w').close()

        original_stdout = sys.stdout  # Save a reference to the original standard output
        with open('log.txt', 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print("Generating report...")
            print(f"Best models: {self.best_models}")
            print(f"Best models scores: {self.best_models_scores}")
            print(f"Best metrics: {self.best_metrics}")
            print("the end")
            print("is working")
            print("is only working when started from terminal")
            sys.stdout = original_stdout  # Reset the standard output to its original value

    def save(self):
        with open(f"{self.path}/medaid.pkl", 'wb') as f:
            pickle.dump(self, f)

