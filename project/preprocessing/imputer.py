import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

class MissingValueImputer:
    def __init__(self, target_column, linear_correlation_threshold=0.8, rf_correlation_threshold=0.2):
        """
        Initialize the imputer.

        Parameters:
        - target_column (str): The name of the target column (Y).
        - linear_correlation_threshold (float): Threshold above which linear regression will be used.
        - rf_correlation_threshold (float): Threshold for moderate correlation to use Random Forest.
        """
        self.target_column = target_column
        self.linear_correlation_threshold = linear_correlation_threshold
        self.rf_correlation_threshold = rf_correlation_threshold

    def impute_missing_values(self, dataframe):
        """
        Impute missing values based on correlation with the target column.

        Parameters:
        - dataframe (pd.DataFrame): The dataframe to process.

        Returns:
        - pd.DataFrame: DataFrame with missing values imputed.
        """
        df_copy = dataframe.copy()

        # Calculate correlation between features and the target column
        correlations = df_copy.corr()[self.target_column]

        for column in df_copy.columns:
            if df_copy[column].isnull().any():  # Check if the column has missing values
                print(f"Imputing missing values in column '{column}'.")

                # For numerical columns
                if df_copy[column].dtype != 'object' or df_copy[column].dtype != 'category':
                    correlation = abs(correlations[column])

                    # If the correlation is strong (linear relationship), use linear regression
                    if correlation >= self.linear_correlation_threshold:
                        print(f"Using Linear Regression for column '{column}'")
                        X = df_copy.dropna(subset=[column])[self.target_column].values.reshape(-1, 1)  # Target variable as independent variable
                        y = df_copy.dropna(subset=[column])[column].values  # Column with missing values as dependent variable

                        # Train linear regression model
                        model = LinearRegression()
                        model.fit(X, y)

                        # Predict missing values
                        missing_values = df_copy[df_copy[column].isnull()]
                        predicted_values = model.predict(missing_values[self.target_column].values.reshape(-1, 1))

                        # Fill missing data
                        df_copy.loc[df_copy[column].isnull(), column] = predicted_values

                    # If the correlation is moderate, use random forest regressor
                    elif correlation >= self.rf_correlation_threshold:
                        print(f"Using Random Forest for column '{column}'")
                        X = df_copy.dropna(subset=[column])[self.target_column].values.reshape(-1, 1)  # Target variable as independent variable
                        y = df_copy.dropna(subset=[column])[column].values  # Column with missing values as dependent variable

                        # Train Random Forest model
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_model.fit(X, y)

                        # Predict missing values
                        missing_values = df_copy[df_copy[column].isnull()]
                        predicted_values = rf_model.predict(missing_values[self.target_column].values.reshape(-1, 1))

                        # Fill missing data
                        df_copy.loc[df_copy[column].isnull(), column] = predicted_values

                    # If the correlation is weak, use mean or median for imputation
                    else:
                        print(f"Using mean/median for column '{column}'")
                        if df_copy[column].dtype == 'object':
                            # Use mode for categorical columns
                            imputed_value = df_copy[column].mode()[0]
                        else:
                            # Use median for numerical columns with weak correlation
                            imputed_value = df_copy[column].median()

                        df_copy[column].fillna(imputed_value, inplace=True)

                # For categorical columns
                else:
                    # If correlation with the column is above the threshold, use decision tree regressor
                    if abs(correlations[column]) >= self.linear_correlation_threshold:
                        print(f"Using Decision Tree Regressor for column '{column}'")
                        X = df_copy.dropna(subset=[column])[self.target_column].values.reshape(-1, 1)  # Target variable as independent variable
                        y = df_copy.dropna(subset=[column])[column].values  # Column with missing values as dependent variable

                        # Train decision tree regressor
                        model = DecisionTreeRegressor()
                        model.fit(X, y)

                        # Predict missing values
                        missing_values = df_copy[df_copy[column].isnull()]
                        predicted_values = model.predict(missing_values[self.target_column].values.reshape(-1, 1))

                        # Fill missing data
                        df_copy.loc[df_copy[column].isnull(), column] = predicted_values
                    else:
                        # If no strong correlation, use mode (most frequent value) for categorical columns
                        imputed_value = df_copy[column].mode()[0]  # Mode for categorical columns
                        print(f"Using mode for column '{column}'")
                        df_copy[column].fillna(imputed_value, inplace=True)

        return df_copy
