import pandas as pd

class ColumnRemover:
    """
    A class to detect and remove non-categorical text columns in a pandas DataFrame,
    including columns with 'id' in their name, highly correlated columns, and more.
    """
    def __init__(self, target_column, threshold=0.2, correlation_threshold=0.9):
        """
        Initialize the detector with thresholds.

        Parameters:
        - threshold (float): The percentage difference of unique values to total values above which a column is considered categorical.
        - correlation_threshold (float): Threshold for correlation between columns.
        """
        self.threshold = threshold
<<<<<<< HEAD
        self.correlation_threshold = correlation_threshold
        self.removal_info = {}
        self.target_column = target_column
=======
        self.removal_info = {}  # Store info about columns and whether they were removed

    def is_categorical(self, column):
        """
        Check if a single column is categorical based on the percentage difference between unique values and total values.

        Parameters:
        - column (pd.Series): The column to check.

        Returns:
        - bool: True if the column is categorical, False otherwise.
        """

        # Check for 'category' dtype or 'object' dtype (typically text data)
        if column.dtype.name == 'category' or column.dtype.name == 'object':
            return True
        return False
>>>>>>> main

    def remove_id_columns(self, dataframe):
        """
        Remove columns that have 'id' in their name (case insensitive).

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - dataframe (pd.DataFrame): The DataFrame with 'id' columns removed.
        """
<<<<<<< HEAD
        id_columns_to_remove = [col for col in dataframe.columns if 'id' in col.lower()]
        for col in id_columns_to_remove:
            if col == self.target_column:
                continue
            self.removal_info[col] = {"Removed": True, "Reason": "Contains 'id'"}
        dataframe.drop(columns=id_columns_to_remove, inplace=True)
        return dataframe
=======
        text_columns_to_drop = []
        for column_name in dataframe.columns:
            if 'id' in column_name.lower():  # Case insensitive search for 'id' in column names
                text_columns_to_drop.append(column_name)
                self.removal_info[column_name] = True  # Mark as removed
        
        return text_columns_to_drop
>>>>>>> main

    def remove_highly_correlated_columns(self, dataframe):
        """
        Identify highly correlated numeric columns and remove one from each pair.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - dataframe (pd.DataFrame): The DataFrame with highly correlated columns removed.
        """

        numeric_columns = dataframe.select_dtypes(include=['number']).columns

        correlation_matrix = dataframe[numeric_columns].corr()

        to_remove = set()

        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > self.correlation_threshold:
                    col_to_remove = correlation_matrix.columns[i]
                    if col_to_remove not in to_remove:
                        if col_to_remove == self.target_column:
                            continue
                        to_remove.add(col_to_remove)
                        self.removal_info[col_to_remove] = {"Removed": True, "Reason": "High correlation"}

        dataframe.drop(columns=to_remove, inplace=True)
        return dataframe


    def remove_non_categorical_text_columns(self, dataframe):
        """
        Remove non-categorical text columns based on the threshold.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - dataframe (pd.DataFrame): The DataFrame with non-categorical text columns removed.
        """
        text_columns_to_drop = []

        for column_name in dataframe.columns:
            column = dataframe[column_name]
            if column.dtype == 'object':  # Text column
                unique_values = column.nunique()
                total_values = len(column)
                percentage_difference = (unique_values / total_values) * 100
                if percentage_difference > self.threshold * 100:
                    if column_name == self.target_column:
                        continue
                    text_columns_to_drop.append(column_name)
                    self.removal_info[column_name] = {"Removed": True, "Reason": "To many unique text values"}
                else:
                    self.removal_info[column_name] = {"Removed": False}

        dataframe.drop(columns=text_columns_to_drop, inplace=True)
        return dataframe

    def remove(self, dataframe):
        """
        Sequentially remove 'id' columns, highly correlated columns, and non-categorical text columns.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - dataframe (pd.DataFrame): The DataFrame after processing.
        """

        for column_name in dataframe.columns:
            self.removal_info[column_name] = {"Removed": False}

        # Step 1: Remove 'id' columns
        dataframe = self.remove_id_columns(dataframe)

        # Step 2: Remove highly correlated columns
        dataframe = self.remove_highly_correlated_columns(dataframe)

        # Step 3: Remove non-categorical text columns
        dataframe = self.remove_non_categorical_text_columns(dataframe)

        return dataframe

    def get_removal_info(self):
        """
        Returns a dictionary with column names and whether they were removed or not.
        """
        return self.removal_info
