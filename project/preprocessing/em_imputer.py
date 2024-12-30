import numpy as np
import pandas as pd

class EMImputer:
    def __init__(self, max_iter=100, tol=1e-4):
        """
        Initialize the EM Imputer.
        Parameters:
            max_iter: Maximum number of iterations for convergence.
            tol: Tolerance for convergence (stop when change < tol).
        """
        self.max_iter = max_iter
        self.tol = tol
        self.mean_ = None
        self.covariance_ = None

    def fit_transform(self, data):
        """
        Perform EM-based imputation on the dataset.
        Parameters:
            data: A pandas DataFrame or numpy array with missing values (NaN).
        Returns:
            Imputed data as a pandas DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            columns = data.columns
            data = data.to_numpy()
        else:
            columns = [f"Var{i}" for i in range(data.shape[1])]

        # Mask missing values
        missing_mask = np.isnan(data)

        # Initialize missing values with column means
        complete_data = data.copy()
        for col in range(data.shape[1]):
            mean = np.nanmean(complete_data[:, col])
            complete_data[missing_mask[:, col], col] = mean

        # Iterative EM algorithm
        for iteration in range(self.max_iter):
            prev_data = complete_data.copy()

            # M-step: Compute mean and covariance
            self.mean_ = np.nanmean(complete_data, axis=0)
            self.covariance_ = np.cov(complete_data, rowvar=False)

            # E-step: Impute missing values
            for row in range(complete_data.shape[0]):
                missing = missing_mask[row]
                if np.any(missing):
                    observed = ~missing
                    mu_obs = self.mean_[observed]
                    mu_mis = self.mean_[missing]
                    cov_obs = self.covariance_[np.ix_(observed, observed)]
                    cov_mis_obs = self.covariance_[np.ix_(missing, observed)]

                    # Conditional mean for missing values
                    mis_values = mu_mis + cov_mis_obs @ np.linalg.inv(cov_obs) @ (complete_data[row, observed] - mu_obs)
                    complete_data[row, missing] = mis_values

            # Check convergence
            change = np.linalg.norm(complete_data - prev_data)
            if change < self.tol:
                print(f"Converged after {iteration + 1} iterations.")
                break

        return pd.DataFrame(complete_data, columns=columns)
