import pandas as pd
from project.preprocessing.preprocessing import Preprocessing

def main():
    data_path = "data/binary/diabetes_dataset.csv" 

    try:
        df = pd.read_csv(data_path)
        print("Original DataFrame:")
        print(df.head())
    except FileNotFoundError:
        print(f"File not found at {data_path}. Please make sure the file exists.")
        return

    target_column = "Outcome" 
    preprocessing = Preprocessing(target_column=target_column)

    try:
        processed_df = preprocessing.preprocess(df)
        print("\nProcessed DataFrame:")
        print(processed_df.head())
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return

    try:
        preprocessing.save_column_info()
        print("\nPreprocessing details saved.")
    except Exception as e:
        print(f"An error occurred while saving preprocessing details: {e}")
        return

if __name__ == "__main__":
    main()
