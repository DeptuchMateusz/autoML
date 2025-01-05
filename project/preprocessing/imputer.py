import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

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

        # Filter only numerical columns for correlation calculation
        numerical_columns = df_copy.select_dtypes(include=['number']).columns

        # Compute correlations only for numerical columns
        correlations = df_copy[numerical_columns].corr()[self.target_column]

        for column in df_copy.columns:
            if df_copy[column].isnull().any():  # Check if the column has missing values

                # For numerical columns
                if column in numerical_columns:
                    correlation = abs(correlations.get(column, 0))  # Get correlation, default to 0 if not found
                    print(f"Korelacja: {correlation}")

                    # If the correlation is strong (linear relationship), use linear regression
                    if correlation >= self.linear_correlation_threshold:
                        print("Regresja")
                        X = df_copy.dropna(subset=[column])[self.target_column].values.reshape(-1, 1)
                        y = df_copy.dropna(subset=[column])[column].values

                        model = LinearRegression()
                        model.fit(X, y)

                        missing_values = df_copy[df_copy[column].isnull()]
                        predicted_values = model.predict(missing_values[self.target_column].values.reshape(-1, 1))

                        df_copy.loc[df_copy[column].isnull(), column] = predicted_values

                    # If the correlation is moderate, use random forest regressor
                    elif correlation >= self.rf_correlation_threshold:
                        print("Las losowy")
                        X = df_copy.dropna(subset=[column])[self.target_column].values.reshape(-1, 1)
                        y = df_copy.dropna(subset=[column])[column].values

                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_model.fit(X, y)

                        missing_values = df_copy[df_copy[column].isnull()]
                        predicted_values = rf_model.predict(missing_values[self.target_column].values.reshape(-1, 1))

                        df_copy.loc[df_copy[column].isnull(), column] = predicted_values

                    # If the correlation is weak, use mean or median for imputation
                    else:
                        print("Mediana")
                        df_copy[column].fillna(df_copy[column].median(), inplace=True)

                    # If the column was integer, cast back to int
                    if df_copy[column].dtype == 'Int64':  
                        df_copy.loc[df_copy[column].isnull(), column] = np.round(predicted_values).astype(int)

                   

                # For categorical columns
                else:
                    print("Decision Tree")
                    # Encode categorical column
                    le = LabelEncoder()
                    non_null_data = df_copy.loc[df_copy[column].notnull(), column]
                    le.fit(non_null_data)
                    df_copy.loc[df_copy[column].notnull(), column] = le.transform(non_null_data)

                    # Preprocessing data for training
                    X = df_copy.dropna(subset=[column])[self.target_column].values.reshape(-1, 1)
                    y = df_copy.dropna(subset=[column])[column].astype(int).values  # Ensure y is int for categorical column

                    # Training the model
                    model = DecisionTreeClassifier(random_state=42)
                    model.fit(X, y)

                    # Predicting missing values
                    missing_values = df_copy[df_copy[column].isnull()]
                    predicted_values = model.predict(missing_values[self.target_column].values.reshape(-1, 1))

                    # Filling missing values in the DataFrame
                    df_copy.loc[df_copy[column].isnull(), column] = predicted_values
                    df_copy[column] = le.inverse_transform(df_copy[column].astype(int))

        return df_copy
