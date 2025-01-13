from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler, GridSearchCV
from itertools import product


class CustomRandomizedSearchCV(RandomizedSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, cv=None,
                 refit=True, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score='raise', return_train_score=False,
                 scoring=None, name=None):
        super().__init__(estimator=estimator,
                         param_distributions=param_distributions,
                         n_iter=n_iter,
                         cv=cv,
                         refit=refit,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         pre_dispatch=pre_dispatch,
                         random_state=random_state,
                         error_score=error_score,
                         return_train_score=return_train_score,
                         scoring=scoring)

        self.results_df = pd.DataFrame()
        self.name = name



    def _run_search(self, evaluate_candidates):
        # Sample parameter combinations
        sampled_params = list(ParameterSampler(self.param_distributions, self.n_iter, random_state=self.random_state))

        # Use tqdm for real-time progress tracking
        with tqdm(total=self.n_iter, desc=f"{self.name} progress", leave=True) as pbar:
            def wrapped_evaluate_candidates(param_iterable):
                for params in param_iterable:
                    evaluate_candidates([params])
                    pbar.update(1)  # Update progress after each evaluation

            # Pass the sampled parameter combinations
            wrapped_evaluate_candidates(sampled_params)

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)


        # Collect results for each iteration
        results = []
        for idx, params in enumerate(self.cv_results_['params']):
            result_entry = {
                'Combination_ID': idx + 1,
            }
            # Add parameter combination
            result_entry.update(params)

            # Add scores for all metrics
            for metric in self.cv_results_.keys():
                if metric.startswith('mean_test_'):
                    metric_name = metric.replace('mean_test_', '')
                    result_entry[metric_name] = self.cv_results_[metric][idx]

            results.append(result_entry)

        self.results_df = pd.DataFrame(results)

        return self

class CustomGridSearchCV(GridSearchCV):
    def __init__(self, estimator, param_grid, cv=None,
                 refit=True, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs',
                 error_score='raise', return_train_score=False,
                 scoring=None, name=None):
        super().__init__(estimator=estimator,
                         param_grid=param_grid,
                         cv=cv,
                         refit=refit,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         pre_dispatch=pre_dispatch,
                         error_score=error_score,
                         return_train_score=return_train_score,
                         scoring=scoring,
                         )
        self.results_df = pd.DataFrame()
        self.name = name

    def _run_search(self, evaluate_candidates):
        # Calculate total combinations in param grid
        param_combinations = list(self._expand_param_grid(self.param_grid))
        total_combinations = len(param_combinations)

        # Use tqdm for progress tracking
        desc = f"{self.name} progress"
        with tqdm(total=total_combinations, desc=desc, leave=True) as pbar:
            def wrapped_evaluate_candidates(param_iterable):
                for params in param_iterable:
                    evaluate_candidates([params])
                    pbar.update(1)  # Update progress after each evaluation

            # Pass the parameter grid combinations
            wrapped_evaluate_candidates(param_combinations)

    def _expand_param_grid(self, param_grid):
        """
        Expands the param_grid into a list of all combinations.
        This handles single dict or list of dicts.
        """
        if isinstance(param_grid, dict):
            keys, values = zip(*param_grid.items())
            for v in product(*values):
                yield dict(zip(keys, v))
        else:  # Assume list of dicts
            for grid in param_grid:
                keys, values = zip(*grid.items())
                for v in product(*values):
                    yield dict(zip(keys, v))

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)

        # Collect results for each iteration
        results = []
        for idx, params in enumerate(self.cv_results_['params']):
            result_entry = {
                'Combination_ID': idx + 1,
            }
            # Add parameter combination
            result_entry.update(params)

            # Add scores for all metrics
            for metric in self.cv_results_.keys():
                if metric.startswith('mean_test_'):
                    metric_name = metric.replace('mean_test_', '')
                    result_entry[metric_name] = self.cv_results_[metric][idx]

            results.append(result_entry)

        self.results_df = pd.DataFrame(results)

        return self
