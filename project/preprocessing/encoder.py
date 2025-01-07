import pandas as pd

class Encoder:
    """
    A class to detect and encode categorical columns in a pandas DataFrame.
    """

    def __init__(self):
        """
        Initialize the Encoder.

        Parameters:
        - target_column (str): The name of the target column to exclude from encoding.
        """
        self.encoding_info = {}

    def is_categorical(self, column):
        """
        Check if a single column is categorical based on dtype.

        Parameters:
        - column (pd.Series): The column to check.

        Returns:
        - bool: True if the column is categorical, False otherwise.
        """
        if not isinstance(column, pd.Series):
            raise ValueError("Input must be a pandas Series.")

        # Check for 'category' dtype or 'object' dtype (typically text data)
        return column.dtype.name in ['category', 'object']

    def encode(self, dataframe, target_column):
        """
        Apply one-hot encoding to categorical columns in the DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to process.

        Returns:
        - pd.DataFrame: A DataFrame with categorical columns one-hot encoded.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        processed_df = dataframe.copy()

        # Find categorical columns excluding the target column
        categorical_columns = [
            col for col in dataframe.columns 
            if self.is_categorical(dataframe[col]) and col != target_column
        ]

        # One-hot encode categorical columns
        for col in categorical_columns:
            unique_values = dataframe[col].nunique()
            self.encoding_info[col] = {
                "Type": "Categorical",
                "Encoded": True,
                "Unique Values": unique_values,
                "Encoding Method": "One-Hot Encoding"
            }

        if categorical_columns:
            processed_df = pd.get_dummies(
                processed_df, 
                columns=categorical_columns, 
                drop_first=True, 
                dtype=int
            )

        return processed_df

    def get_encoding_info(self):
        """
        Get details about the encoding process.

        Returns:
        - dict: A dictionary containing information about the encoded columns.
        """
        return self.encoding_info
