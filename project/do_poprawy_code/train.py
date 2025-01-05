from search import CustomRandomizedSearchCV, CustomGridSearchCV
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd

# TODO: co zrobic z mode?
# TODO: siatki parametrów



def train(X, y, models, metric, mode, path, search, cv, n_iter):
    param_grids = {
        "logistic": {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear',
                      # 'saga' # ten dziad jest wkurzajacy: wykorzystuje ciagle max_iter
                       ],
            #'max_iter': [100]
        },
        "tree": {
            'max_depth': [3, 5, 7, 9, 11, 13, 15],
            'min_samples_split': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        },
        "random_forest": {
            'n_estimators': [100, 300, 400],
            'max_depth': [3, 5, 8, 11, 15],
            'min_samples_split': [2, 4, 6, 8],
            'min_samples_leaf': [1, 3, 5]
        },
        "xgboost": {
            'verbosity': [0],
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
        'verbosity': [-1],
        'learning_rate': [0.005, 0.01],
        'n_estimators': [8,16,24],
        'num_leaves': [6,8,12,16],
        'boosting_type' : ['gbdt', 'dart'],
        'objective' : ['binary'],
        'max_bin':[255, 510],
        'random_state' : [500],
        'colsample_bytree' : [0.64, 0.65, 0.66],
        'subsample' : [0.7,0.75],
        'reg_alpha' : [1,1.2],
        'reg_lambda' : [1,1.2,1.4],
        }
    }

    #create results folder
    if not os.path.exists(f"{path}/results"):
        os.makedirs(f"{path}/results")

    best_models = []
    best_models_scores = []
    metrics_list = []

    for model in models:
        param_grid = param_grids[model]
        if model == "logistic":
            model_with_params = LogisticRegression()
        elif model == "tree":
            model_with_params = DecisionTreeClassifier()
        elif model == "random_forest":
            model_with_params = RandomForestClassifier()
        elif model == "xgboost":
            model_with_params = XGBClassifier()
        elif model == "lightgbm":
            model_with_params = LGBMClassifier()
        if search == "random":
            rs = CustomRandomizedSearchCV(model_with_params, param_grid, n_iter=n_iter, cv=cv,
                                          scoring={'f1': 'f1', 'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall'},
                                          refit = metric, name=model)
        else:
            rs = CustomGridSearchCV(model_with_params, param_grid,  cv=cv,
                                          scoring={'f1': 'f1', 'accuracy': 'accuracy', 'precision': 'precision', 'recall': 'recall'},
                                          refit = metric, name=model)

        rs.fit(X, y)
        rs.results_df.to_csv(f"{path}/results/{model}.csv", index=False)
        best_models.append(rs.best_estimator_)
        best_models_scores.append(rs.best_score_)

        best_metrics = {
            'model': model,
            'best_score': rs.best_score_,
            'f1': rs.cv_results_['mean_test_f1'][rs.best_index_],
            'accuracy': rs.cv_results_['mean_test_accuracy'][rs.best_index_],
            'precision': rs.cv_results_['mean_test_precision'][rs.best_index_],
            'recall': rs.cv_results_['mean_test_recall'][rs.best_index_]
        }
        metrics_list.append(best_metrics)

        best_models = [x for _, x in sorted(zip(best_models_scores, best_models), reverse=True)]
        best_models_scores = sorted(best_models_scores, reverse=True)
        metrics_df = pd.DataFrame(metrics_list).sort_values(by='best_score', ascending=False)

    return best_models, best_models_scores, metrics_df

