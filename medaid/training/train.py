<<<<<<< HEAD
<<<<<<< HEAD
from charset_normalizer import is_binary
=======
>>>>>>> ed71fee (male poprawki)
=======
from charset_normalizer import is_binary
>>>>>>> a88fd18 (wtorek)
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer, accuracy_score

from medaid.training.search import CustomRandomizedSearchCV, CustomGridSearchCV
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import get_scorer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*ConvergenceWarning.*")



# TODO: siatki parametrów



def train(X, y, X_test, y_test, models, metric, path, search, cv, n_iter):
    warnings.filterwarnings("ignore", category=UserWarning, message=".*ConvergenceWarning.*")
<<<<<<< HEAD

    number_of_classes = len(y.unique()) if len(y.unique()) > 2 else 1
=======
>>>>>>> ed71fee (male poprawki)

    number_of_classes = len(y.unique()) if len(y.unique()) > 2 else 1

    param_grids = {
        "logistic": {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l2'],
            'solver': [
                       'saga',
                       'lbfgs',
                        'newton-cg'
                       ],
            #'max_iter': [100]
        },
        "tree": {
            'max_depth': [3, 5, 7, 9, 11, 13, 15],
            'min_samples_split': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5]
        },
        "random_forest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 8, 11],
            'min_samples_split': [2, 4, 6, 8],
            'min_samples_leaf': [1, 3, 5],
            # 'n_jobs': [-1]
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
            'gamma': [0, 0.1, 0.5, 1, 10],
            # 'n_jobs': [-1]
        },
        "lightgbm": {
        'verbosity': [-1],
        'learning_rate': [0.005, 0.01],
        'n_estimators': [8,16,24],
        'num_leaves': [6,8,12,16],
        'boosting_type' : ['gbdt', 'dart'],
        'max_bin':[255, 510],
        'random_state' : [500],
        'colsample_bytree' : [0.5, 0.6, 0.7],
        'subsample' : [0.7,0.75],
        'reg_alpha' : [1,1.2],
        'reg_lambda' : [1,1.2,1.4],
        }
    }


    best_models = []
    best_models_scores = []
    metrics_list = []

    for model in models:
        param_grid = param_grids[model]
        if model == "logistic":
            model_with_params = LogisticRegression(n_jobs=-1, max_iter=1000)
        elif model == "tree":
            model_with_params = DecisionTreeClassifier()
        elif model == "random_forest":
            model_with_params = RandomForestClassifier(n_jobs=-1)
        elif model == "xgboost":
            model_with_params = XGBClassifier(n_jobs=-1)
        elif model == "lightgbm":
<<<<<<< HEAD
<<<<<<< HEAD
            model_with_params = LGBMClassifier(n_jobs=-1, objective='binary' if number_of_classes == 1 else 'multiclass', num_class=number_of_classes)
=======
            model_with_params = LGBMClassifier(n_jobs=-1)
>>>>>>> ed71fee (male poprawki)
=======
            model_with_params = LGBMClassifier(n_jobs=-1, objective='binary' if number_of_classes == 1 else 'multiclass', num_class=number_of_classes)
>>>>>>> a88fd18 (wtorek)
        if search == "random":
            rs = CustomRandomizedSearchCV(model_with_params, param_grid, n_iter=n_iter, cv=cv,
                                          scoring={'f1': make_scorer(f1_score, average='weighted'),
                                                   'accuracy': make_scorer(accuracy_score),
                                                   'precision': make_scorer(precision_score, average='weighted'),
                                                   'recall': make_scorer(recall_score, average='weighted')},
                                          refit = metric, name=model,
                                          n_jobs=-1)
        else:
            rs = CustomGridSearchCV(model_with_params, param_grid,  cv=cv,
                                          scoring={'f1': make_scorer(f1_score, average='weighted'),
                                                   'accuracy': make_scorer(accuracy_score),
                                                   'precision': make_scorer(precision_score, average='weighted'),
                                                   'recall': make_scorer(recall_score, average='weighted')},
                                          refit = metric, name=model)

        rs.fit(X, y)
        rs.results_df.to_csv(f"{path}/results/models/{model}.csv", index=False)
        best_models.append(rs.best_estimator_)
        best_models_scores.append(rs.best_score_)


        # scorer = get_scorer(metric)
        # create with make_scorer depending on metric = f1, accuracy, precision, recall, with average='weighted'
        metrics_dict = {
            'f1': make_scorer(f1_score, average='weighted'),
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall': make_scorer(recall_score, average='weighted')
        }
        scorer = metrics_dict[metric]


        test_best_score = scorer(rs.best_estimator_, X_test, y_test)
        best_metrics = {
            'model': model,
            'best_score': rs.best_score_,
            'f1': rs.cv_results_['mean_test_f1'][rs.best_index_],
            'accuracy': rs.cv_results_['mean_test_accuracy'][rs.best_index_],
            'precision': rs.cv_results_['mean_test_precision'][rs.best_index_],
            'recall': rs.cv_results_['mean_test_recall'][rs.best_index_],
            'test_best_score': test_best_score,
            'test_f1': f1_score(y_test, rs.best_estimator_.predict(X_test), average='weighted'),
            'test_accuracy': accuracy_score(y_test, rs.best_estimator_.predict(X_test)),
            'test_precision': precision_score(y_test, rs.best_estimator_.predict(X_test), average='weighted'),
            'test_recall': recall_score(y_test, rs.best_estimator_.predict(X_test), average='weighted')
        }
        metrics_list.append(best_metrics)

        best_models_scores, best_models = zip(
            *sorted(zip(best_models_scores, best_models), key=lambda pair: pair[0], reverse=True)
        )
        best_models = list(best_models)
        best_models_scores = list(best_models_scores)
        metrics_df = pd.DataFrame(metrics_list).sort_values(by='best_score', ascending=False)

    return best_models, best_models_scores, metrics_df
