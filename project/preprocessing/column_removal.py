import pandas as pd

class ColumnRemover:
    """
    A class to detect and remove non-categorical text columns in a pandas DataFrame,
    including columns with 'id' in their name.
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

    def remove_id_columns(self, dataframe):
        """
        Remove columns that have 'id' in their name (case insensitive).

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - list: List of column names that were removed.
        """
        text_columns_to_drop = []
        for column_name in dataframe.columns:
            if 'id' in column_name.lower():  # Case insensitive search for 'id' in column names
                text_columns_to_drop.append(column_name)
                self.removal_info[column_name] = True  # Mark as removed
        
        return text_columns_to_drop

    def remove(self, dataframe):
        """
        Delete non-categorical text columns and columns containing 'id' in their name.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - pd.DataFrame: A DataFrame with non-categorical text columns removed.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        processed_df = dataframe.copy()
        text_columns_to_drop = []

        # First, remove 'id' columns
        text_columns_to_drop.extend(self.remove_id_columns(dataframe))

        # Now check for other non-categorical text columns
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

        # Drop the identified columns
        processed_df.drop(columns=text_columns_to_drop, inplace=True)

        return processed_df

    def get_removal_info(self):
        """
        Returns a dictionary with column names and whether they were removed or not.
        """
        return self.removal_info
