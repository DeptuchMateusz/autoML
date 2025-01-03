import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from em_imputer import EMImputer

class ImputationStrategies:
    @staticmethod
    def get_methods():
        """Returns a dictionary of all available imputers."""
        return {
            "Mean": SimpleImputer(strategy="mean"),
            "Median": SimpleImputer(strategy="median"),
            "KNN": KNNImputer(n_neighbors=5),
            "Iterative": IterativeImputer(),
            "EM": EMImputer()
        }