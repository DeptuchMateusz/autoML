import unittest
import pandas as pd
from project.preprocessing.categorical_column_handler import CategoricalColumnHandler

class TestCategoricalColumnHandler(unittest.TestCase):

    # This method will run before each test
    def setUp(self):
        # Initialize the CategoricalColumnHandler (detector)
        self.detector = CategoricalColumnHandler()

        # Example DataFrame for testing
        self.df = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Female', 'Male'],
            'Age': [23, 25, 30, 22],
            'Country': ['US', 'UK', 'CA', 'US'],
            'Score': [85, 90, 88, 92],
            'Comments': ['Good', 'Very Good', 'Average', 'Excellent']
        })

    def test_is_categorical(self):
        # Testing if the function correctly identifies categorical columns
        self.assertTrue(self.detector.is_categorical(self.df['Gender']))  # Gender is categorical
        self.assertFalse(self.detector.is_categorical(self.df['Age']))  # Age is numerical
        self.assertTrue(self.detector.is_categorical(self.df['Country']))  # Country is categorical
        self.assertFalse(self.detector.is_categorical(self.df['Score']))  # Score is numerical

    def test_detect_categorical_columns(self):
        # Testing if the function detects all categorical columns correctly
        categorical_columns = self.detector.detect_categorical_columns(self.df)

        # Gender and Country are categorical columns
        self.assertIn('Gender', categorical_columns)
        self.assertIn('Country', categorical_columns)
        
        # Age and Score are not categorical
        self.assertNotIn('Age', categorical_columns)
        self.assertNotIn('Score', categorical_columns)

    def test_filter_and_encode(self):
        # Testing filter_and_encode function
        processed_df = self.detector.filter_and_encode(self.df)

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
    unittest.main()
