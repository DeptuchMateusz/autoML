import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Scaler:
    """
    A class that scales features in a DataFrame based on their distribution (normalization or standardization).
    """
    def __init__(self, target_column):
        self.target_column = target_column
        self.scaling_params = {}  # Store scaling parameters (scaling_method, params) for each column

    def _detect_distribution(self, column):
        """
        Detect the distribution of the data (skewed or normal).
        - If skewness is close to 0, assume normal distribution.
        - Otherwise, assume skewed distribution.
        """
        col_skew = skew(column.dropna())  # Calculate skewness
        return 'normal' if abs(col_skew) < 0.3 else 'skewed'

    def normalize(self, column):
        """
        Normalize the data between 0 and 1 (Min-Max scaling).
        This requires saving min and max for later inverse transformation.
        """
        scaler = MinMaxScaler()
        reshaped_column = column.values.reshape(-1, 1)
        normalized_column = scaler.fit_transform(reshaped_column)

        # Save the scaling method and relevant parameters
        self.scaling_params[column.name] = {
            'scaling_method': 'min_max',
            'params': {
                'min': scaler.data_min_[0],
                'max': scaler.data_max_[0]
            }
        }
        return normalized_column.flatten()

    def standardize(self, column):
        """
        Standardize the data (z-score transformation).
        This requires saving mean and std for later inverse transformation.
        """
        scaler = StandardScaler()
        reshaped_column = column.values.reshape(-1, 1)
        standardized_column = scaler.fit_transform(reshaped_column)

        # Save the scaling method and relevant parameters
        self.scaling_params[column.name] = {
            'scaling_method': 'standardization',
            'params': {
                'mean': scaler.mean_[0],
                'std': scaler.scale_[0]
            }
        }
        return standardized_column.flatten()

    def scale_column(self, column):
        """
        Scales the given column based on its distribution.
        - If the column is normally distributed, standardize it.
        - If the column is skewed, normalize it.
        """
        distribution_type = self._detect_distribution(column)

        if distribution_type == 'normal':
            return self.standardize(column)
        else:
            return self.normalize(column)

    def scale(self, dataframe):
        """
        Iteratively scale each numeric column in the DataFrame.
        
        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to scale.
        
        Returns:
        - pd.DataFrame: A DataFrame with scaled numeric columns.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        scaled_df = dataframe.copy()

        for column_name in dataframe.select_dtypes(include=[np.number]).columns:
            if column_name == self.target_column:
                continue
            scaled_df[column_name] = self.scale_column(dataframe[column_name])

        return scaled_df

    def get_scaling_info(self):
        """
        Return the scaling information for all columns in the dataframe.
        This will include the scaling method (min_max or standardization),
        and the parameters used for scaling (min, max, mean, std).
        """
        return self.scaling_params
