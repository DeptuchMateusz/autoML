import unittest
import pandas as pd
import sys
from project.preprocessing.categorical_column_handler import CategoricalColumnHandler

class TestCategoricalColumnHandler(unittest.TestCase):

    # This method will run before each test
    def setUp(self):
        # Initialize the CategoricalColumnHandler (detector)
        self.detector = CategoricalColumnHandler(threshold=0.8)

        # Example DataFrame for testing
        self.df = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Female', 'Male'],
            'Age': [23, 25, 30, 22],
            'Score': [85, 90, 88, 92],
            'Country': ['US', 'UK', 'CA', 'US'],
            'Comments': ['Good', 'Very Good', 'Average', 'Excellent']
        })
    

    def test_is_categorical(self):
        # Testing if the function correctly identifies categorical columns
        self.assertTrue(self.detector.is_categorical(self.df['Gender']))  # Gender is categorical
        self.assertFalse(self.detector.is_categorical(self.df['Age']))  # Age is numerical
        self.assertFalse(self.detector.is_categorical(self.df['Score']))  # Score is numerical
        self.assertTrue(self.detector.is_categorical(self.df['Country']))  # Country is categorical
        self.assertTrue(self.detector.is_categorical(self.df['Comments']))  # Comment is categorical


    def test_filter_and_encode(self):
        # Testing filter_and_encode function
        processed_df = self.detector.filter_and_encode(self.df)
 
        print(processed_df.head())

        # Check that 'Comments' column is removed (it should not be categorical)
        self.assertNotIn('Comments', processed_df.columns)

        # Check that categorical columns are one-hot encoded
        self.assertIn('Gender_Male', processed_df.columns)
        self.assertIn('Gender_Female', processed_df.columns)
        self.assertIn('Country_US', processed_df.columns)
        self.assertIn('Country_UK', processed_df.columns)
        self.assertIn('Country_CA', processed_df.columns)

        # Check that non-categorical columns are kept
        self.assertIn('Age', processed_df.columns)
        self.assertIn('Score', processed_df.columns)


if __name__ == '__main__':
    unittest.main(buffer=True)
