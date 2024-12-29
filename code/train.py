from randomSearch import CustomRandomizedSearchCV
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# TODO: co zrobic z mode?
# TODO: siatki parametrów



def train(X, y, models, metric, mode):
    n_iter = 5
    cv = 2
    param_grids = {
        "logistic": {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        "tree": {
            'max_depth': [3, 5, 7, 9, 11, 13, 15],
            'min_samples_split': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        },
        "random_forest": {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 5, 7, 9, 11, 13, 15],
            'min_samples_split': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        },
        "xgboost": {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 5, 7, 9, 11, 13, 15],
            'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7, 1],
            'subsample': [0.5, 0.7, 1],
            'colsample_bytree': [0.5, 0.7, 1],
            'colsample_bylevel': [0.5, 0.7, 1],
            'colsample_bynode': [0.5, 0.7, 1],
            'reg_alpha': [0, 0.1, 0.5, 1, 10],
            'reg_lambda': [0, 0.1, 0.5, 1, 10],
            'gamma': [0, 0.1, 0.5, 1, 10]
        },
        "lightgbm": {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 5, 7, 9, 11, 13, 15],
            'learning_rate': [0.01, 0.1, 0.3, 0.5, 0.7, 1],
            'subsample': [0.5, 0.7, 1],
            'colsample_bytree': [0.5, 0.7, 1],
            'colsample_bylevel': [0.5, 0.7, 1],
            'colsample_bynode': [0.5, 0.7, 1],
            'reg_alpha': [0, 0.1, 0.5, 1, 10],
            'reg_lambda': [0, 0.1, 0.5, 1, 10],
            'gamma': [0, 0.1, 0.5, 1, 10]
        }
    }

    #create results folder
    if not os.path.exists("results"):
        os.makedirs("results")

    best_models = []
    best_models_scores = []

    for i in tqdm(range(len(models))):
        model = models[i]
        param_grid = param_grids[model]
        if model == "logistic":
            model_with_params = LogisticRegression()
        elif model == "tree":
            model_with_params = DecisionTreeClassifier()
        elif model == "random_forest":
            model_with_params = RandomForestClassifier()
        elif model == "xgboost":
            model_with_params = XGBClassifier(verbose = 0)
        elif model == "lightgbm":
            model_with_params = LGBMClassifier(verbose = 0)

        rs = CustomRandomizedSearchCV(model_with_params, param_grid, n_iter=n_iter, cv=cv, scoring=metric)
        rs.fit(X, y)
        print(f"{model}: {rs.best_score_}")
        rs.results_df.to_csv(f"results/{model}.csv", index=False)
        best_models.append(rs.best_estimator_)
        best_models_scores.append(rs.best_score_)
    return best_models, best_models_scores

