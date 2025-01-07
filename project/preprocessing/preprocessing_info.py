import pandas as pd

class PreprocessingCsv:
    """
    A class that collects preprocessing details (removal, imputation, encoding, scaling)
    and exports these details to a CSV file.
    """
    def __init__(self, output_file="project/do_poprawy_code/medaid/results/preprocessing_details.csv"):
        """
        Initialize the exporter with an output file name.

        Parameters:
        - output_file (str): The name of the output CSV file.
        """
        self.output_file = output_file

    def export_to_csv(self, text_column_removal_info, imputation_info, encoding_info, scaling_info):
        """
        Export preprocessing details (text removal, imputation, encoding, scaling) to a CSV file.

        Parameters:
        - text_column_removal_info (dict): Information about text column removal.
        - imputation_info (dict): Information about imputation.
        - encoding_info (dict): Information about encoding.
        - scaling_info (dict): Information about scaling.
        
        Returns:
        - None
        """
        if not isinstance(text_column_removal_info, dict):
            raise ValueError("text_column_removal_info must be a dictionary.")
        if not isinstance(imputation_info, dict):
            raise ValueError("imputation_info must be a dictionary.")
        if not isinstance(encoding_info, dict):
            raise ValueError("encoding_info must be a dictionary.")
        if not isinstance(scaling_info, dict):
            raise ValueError("scaling_info must be a dictionary.")
        
        # Collect unique column names from all preprocessing steps
        all_columns = set(
            text_column_removal_info.keys()
        ).union(imputation_info.keys(), encoding_info.keys(), scaling_info.keys())
        
        # Prepare a list of dictionaries with details for each column

        columns_info = []
        for column in all_columns:
            columns_info.append({
                "Column Name": column,
                "Removed": text_column_removal_info.get(column, None),
                "Imputation Method": imputation_info.get(column, {}).get("Imputation Method", None),
                "Encoded": encoding_info.get(column, {}).get("Encoding Method", None),
                "Scaling Method": scaling_info.get(column, {}).get("scaling_method", None),
                "Scaling Params": scaling_info.get(column, {}).get("params", None),
            })
        # Convert the list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(columns_info)
        print(df.head())

        # Save the DataFrame to a CSV file
        df.to_csv(self.output_file, index=False)
        print(f"CSV file saved as {self.output_file}")
