from charset_normalizer import is_binary
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
from sklearn.exceptions import ConvergenceWarning





# TODO: siatki parametrów
<<<<<<< HEAD
=======



>>>>>>> 9198a1d (report fixes)
def train(X, y, X_test, y_test, models, metric, path, search, cv, n_iter, n_jobs, param_grids):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    number_of_classes = len(y.unique()) if len(y.unique()) > 2 else 1



    best_models = []
    best_models_scores = []
    metrics_list = []

    for model in models:
        param_grid = param_grids[model]
        if model == "logistic":
            model_with_params = LogisticRegression(n_jobs=n_jobs, max_iter=10000)
        elif model == "tree":
            model_with_params = DecisionTreeClassifier()
        elif model == "random_forest":
            model_with_params = RandomForestClassifier(n_jobs=n_jobs)
        elif model == "xgboost":
            model_with_params = XGBClassifier(n_jobs=n_jobs)
        elif model == "lightgbm":
            model_with_params = LGBMClassifier(n_jobs=n_jobs, objective='binary' if number_of_classes == 1 else 'multiclass', num_class=number_of_classes)
        if search == "random":
            rs = CustomRandomizedSearchCV(model_with_params, param_grid, n_iter=n_iter, cv=cv,
                                          scoring={'f1': make_scorer(f1_score, average='weighted'),
                                                   'accuracy': make_scorer(accuracy_score),
                                                   'precision': make_scorer(precision_score, average='weighted', zero_division=0),
                                                   'recall': make_scorer(recall_score, average='weighted', zero_division=0)},
                                          refit = metric, name=model,
                                          n_jobs=n_jobs)
        else:
            rs = CustomGridSearchCV(model_with_params, param_grid,  cv=cv,
                                          scoring={'f1': make_scorer(f1_score, average='weighted'),
                                                   'accuracy': make_scorer(accuracy_score),
                                                   'precision': make_scorer(precision_score, average='weighted', zero_division=0),
                                                   'recall': make_scorer(recall_score, average='weighted', zero_division=0)},
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
            'recall': make_scorer(recall_score, average='weighted', zero_division=0)
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
            'test_precision': precision_score(y_test, rs.best_estimator_.predict(X_test), average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, rs.best_estimator_.predict(X_test), average='weighted', zero_division=0)
        }
        metrics_list.append(best_metrics)

        best_models_scores, best_models = zip(
            *sorted(zip(best_models_scores, best_models), key=lambda pair: pair[0], reverse=True)
        )
        best_models = list(best_models)
        best_models_scores = list(best_models_scores)
        metrics_df = pd.DataFrame(metrics_list).sort_values(by='best_score', ascending=False)

    return best_models, best_models_scores, metrics_df
