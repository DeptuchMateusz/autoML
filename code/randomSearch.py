import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

class CustomRandomizedSearchCV(RandomizedSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10, cv=None,
                 refit=True, n_jobs=None, verbose=0, pre_dispatch='2*n_jobs',
                 random_state=None, error_score='raise', return_train_score=False,
                 dataset_name=None, scoring = 'roc_auc'):
        # Explicitly calling the parent class constructor with named parameters
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
                         return_train_score=return_train_score)

        self.dataset_name = dataset_name
        self.results_df = pd.DataFrame()

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)

        # Collect results for each iteration
        results = []
        for idx, params in enumerate(self.cv_results_['params']):
            score = self.cv_results_['mean_test_score'][idx]
            params['Combination_ID'] = idx + 1
            params['Score'] = score
            params['Dataset'] = self.dataset_name
            results.append(params)

        # Create a DataFrame from the results
        self.results_df = pd.DataFrame(results)

        return self