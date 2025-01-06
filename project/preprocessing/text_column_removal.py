import pandas as pd

class TextColumnRemover:
    """
    A class to detect and remove non-categorical text columns in a pandas DataFrame.
    """

    def __init__(self, threshold=0.2):
        """
        Initialize the detector with thresholds.

        Parameters:
        - threshold (float): The percentage difference of unique values to total values above which a column is considered categorical.
        """
        self.threshold = threshold
        self.removal_info = {}  # Store info about columns and whether they were removed

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

        # Check for 'category' dtype or 'object' dtype (typically text data)
        if column.dtype.name == 'category' or column.dtype.name == 'object':
            return True
        return False

    def remove(self, dataframe):
        """
        Delete non-categorical text columns.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - pd.DataFrame: A DataFrame with non-categorical text columns removed.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        processed_df = dataframe.copy()
        text_columns_to_drop = []

        for column_name in dataframe.columns:
            column = dataframe[column_name]

            if self.is_categorical(column):
                unique_values = column.nunique()
                total_values = len(column)
                percentage_difference = (unique_values / total_values) * 100

                # If the column has more unique values than threshold, mark for removal
                if percentage_difference > self.threshold * 100:
                    text_columns_to_drop.append(column_name)
                    self.removal_info[column_name] = True  # Mark as removed
                else:
                    self.removal_info[column_name] = False  # Not removed
            else:
                self.removal_info[column_name] = False  # Not removed

        # Drop non-categorical text columns
        processed_df.drop(columns=text_columns_to_drop, inplace=True)

        return processed_df

    def get_removal_info(self):
        """
        Returns a dictionary with column names and whether they were removed or not.
        """
        return self.removal_info
