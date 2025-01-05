import pandas as pd

class CategoricalColumnHandler:
    """
    A class to detect categorical columns in a pandas DataFrame.
    """

    def __init__(self, threshold=0.1):
        """
        Initialize the detector with thresholds.

        Parameters:
        - threshold (float): The ratio of unique values to total values below which a column is considered categorical.
        - max_unique_values (int): Maximum number of unique values for a column to be considered categorical.
        """
        self.threshold = threshold
        #self.max_unique_values = max_unique_values

    def is_categorical(self, column):
        """
        Check if a single column is categorical based on thresholds.

        Parameters:
        - column (pd.Series): The column to check.

        Returns:
        - bool: True if the column is categorical, False otherwise.
        """
        if not isinstance(column, pd.Series):
            raise ValueError("Input must be a pandas Series.")

        # Check data type
        if column.dtype == 'object' or column.dtype.name == 'category':
            return True

        # Handle numeric columns
        if pd.api.types.is_numeric_dtype(column):
            unique_values = column.nunique()
            total_values = len(column)
            ratio = unique_values / total_values

            return ratio < self.threshold

        return False

    def detect_categorical_columns(self, dataframe):
        """
        Detect all categorical columns in a DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to analyze.

        Returns:
        - List[str]: A list of column names that are likely categorical.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        categorical_columns = []

        for column_name in dataframe.columns:
            column = dataframe[column_name]
            if self.is_categorical(column):
                categorical_columns.append(column_name)

        return categorical_columns

    def filter_and_encode(self, dataframe):
        """
        Filter out non-categorical text columns and apply one-hot encoding to categorical text columns.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - pd.DataFrame: A DataFrame with non-categorical text columns removed and categorical text columns one-hot encoded.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        processed_df = dataframe.copy()
        text_columns_to_drop = []
        text_columns_to_encode = []

        for column_name in dataframe.columns:
            column = dataframe[column_name]

            # Identify non-categorical text columns
            if column.dtype == 'object' and not self.is_categorical(column):
                text_columns_to_drop.append(column_name)

            # Identify categorical text columns for one-hot encoding
            elif column.dtype == 'object' and self.is_categorical(column):
                text_columns_to_encode.append(column_name)

        # Drop non-categorical text columns
        processed_df.drop(columns=text_columns_to_drop, inplace=True)

        # One-hot encode categorical text columns
        if text_columns_to_encode:
            processed_df = pd.get_dummies(processed_df, columns=text_columns_to_encode, drop_first=True, dtype=int)

        return processed_df

