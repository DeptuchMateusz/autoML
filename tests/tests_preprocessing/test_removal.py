import pandas as pd
from project.preprocessing.column_removal import ColumnRemover  # Adjust the import based on your file structure

# Create a sample DataFrame with columns, including 'id' columns and non-categorical ones
data_path = "data/binary/cardio_train.csv" 

try:
    df = pd.read_csv(data_path, sep=";")
    print("Original DataFrame:")
    print(df.head())
except FileNotFoundError:
    print(f"File not found at {data_path}. Please make sure the file exists.")



# Print the original DataFrame
print("Original DataFrame:")
print(df)

# Initialize the ColumnRemover
column_remover = ColumnRemover(threshold=0.2)

# Apply the remove function to remove 'id' columns and non-categorical columns
processed_df = column_remover.remove(df)

# Print the processed DataFrame
print("\nProcessed DataFrame (after removing 'id' and non-categorical columns):")
print(processed_df)

# Print removal information
print("\nRemoval Information:")
removal_info = column_remover.get_removal_info()
print(removal_info)
