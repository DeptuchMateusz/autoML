import pandas as pd

class CategoricalColumnHandler:
    """
    A class to detect categorical columns in a pandas DataFrame.
    """

    def __init__(self, threshold=0.2):
        """
        Initialize the detector with thresholds.

        Parameters:
        - threshold (float): The percentage difference of unique values to total values above which a column is considered categorical.
        """
        self.threshold = threshold

    def is_categorical(self, column):
        """
        Check if a single column is categorical based on the percentage difference between unique values and total values.

        Parameters:
        - column (pd.Series): The column to check.

        Returns:
        - bool: True if the column is categorical, False otherwise.
        """
        if not isinstance(column, pd.Series):
            raise ValueError("Input must be a pandas Series.")

        # Check for 'category' dtype
        if column.dtype.name == 'category' or column.dtype.name == 'object':
            return True


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

            if self.is_categorical(column):
                unique_values = column.nunique()
                total_values = len(column)
                percentage_difference = (unique_values / total_values) * 100
                if percentage_difference <= self.threshold * 100:
                    text_columns_to_encode.append(column_name)
                else:
                    text_columns_to_drop.append(column_name)

                
        # Drop non-categorical text columns
        processed_df.drop(columns=text_columns_to_drop, inplace=True)

        # One-hot encode categorical text columns
        if text_columns_to_encode:
            processed_df = pd.get_dummies(processed_df, columns=text_columns_to_encode, drop_first=True, dtype=int)

        return processed_df
