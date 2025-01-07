import pandas as pd
from project.preprocessing.column_removal import ColumnRemover
from project.preprocessing.encoder import Encoder
from project.preprocessing.scaler import Scaler
from project.preprocessing.imputer import Imputer
from project.preprocessing.preprocessing_info import PreprocessingCsv  
from project.preprocessing.numeric_format_handler import NumericCommaHandler

class Preprocessing:
    def __init__(self, target_column, path):
        """
        Initialize the preprocessing pipeline.

        Parameters:
        - target_column (str): The name of the target column that will not be preprocessed.
        - output_file (str): The name of the output CSV file where details will be saved.
        """
        self.target_column = target_column
<<<<<<< HEAD
        self.numeric_format_handler = NumericCommaHandler()
        self.text_column_remover = ColumnRemover(self.target_column)
        self.encoder = Encoder(self.target_column)
        self.scaler = Scaler(self.target_column)
        self.imputation = Imputer(self.target_column)
        self.path = path + "/results/preprocessing_details.csv"
        self.preprocessing_info = PreprocessingCsv(self.path)
=======
        self.text_column_remover = ColumnRemover()
        self.encoder = Encoder()
        self.scaler = Scaler()
        self.imputation = Imputer()
        self.preprcoseesing_info = PreprocessingCsv(output_file)
>>>>>>> 455bbb0 (Revert "Karolina")
        self.columns_info = []  # List to store details of the preprocessing steps

    def preprocess(self, dataframe):
        """
        Run the entire preprocessing pipeline on the provided DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame to preprocess.

        Returns:
        - pd.DataFrame: The preprocessed DataFrame.
        """
<<<<<<< HEAD

        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        # 1. Handle numeric format
        dataframe = self.numeric_format_handler.handle_numeric_format(dataframe)

        # 2. Remove text columns
        dataframe = self.text_column_remover.remove(dataframe)
        text_column_removal_info = self.text_column_remover.get_removal_info()

        # 3. Impute missing values
        dataframe = self.imputation.impute_missing_values(dataframe)
        imputation_info = self.imputation.get_imputation_info()

        # 4. Encode categorical variables
        dataframe = self.encoder.encode(dataframe)
        encoding_info = self.encoder.get_encoding_info()

        # 5. Scale numerical features
        dataframe = self.scaler.scale(dataframe)
=======
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # 1. Remove text columns
        dataframe = self.text_column_remover.remove(dataframe)
        text_column_removal_info = self.text_column_remover.get_removal_info()

        # 2. Impute missing values
        dataframe = self.imputation.impute_missing_values(dataframe, self.target_column)
        imputation_info = self.imputation.get_imputation_info()

        # 3. Encode categorical variables
        dataframe = self.encoder.encode(dataframe, self.target_column)
        encoding_info = self.encoder.get_encoding_info()

        # 4. Scale numerical features
        dataframe = self.scaler.scale(dataframe, self.target_column)
>>>>>>> 455bbb0 (Revert "Karolina")
        scaling_info = self.scaler.get_scaling_info()

        # Save the column info to CSV
        self.save_column_info(text_column_removal_info, imputation_info, encoding_info, scaling_info)

        return dataframe

    def get_column_info(self):
        """
        Retrieve the details of each preprocessing step for all columns.

        Returns:
        - list: Contains details of the preprocessing for each column.
        """
        return self.columns_info

    def save_column_info(self, text_column_removal_info, imputation_info, encoding_info, scaling_info):
        """
        Save the preprocessing details to a CSV file using PreprocessingCsvExporter.

        Parameters:
        - text_column_removal_info (dict): Information about text column removal.
        - imputation_info (dict): Information about imputation.
        - encoding_info (dict): Information about encoding.
        - scaling_info (dict): Information about scaling.
        """
        
        # Use PreprocessingCsvExporter to save the details
<<<<<<< HEAD
        self.preprocessing_info.export_to_csv(text_column_removal_info, imputation_info, encoding_info, scaling_info)
=======
        self.preprcoseesing_info.export_to_csv(text_column_removal_info, imputation_info, encoding_info, scaling_info)
>>>>>>> 455bbb0 (Revert "Karolina")
