import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from project.preprocessing.imputer import MissingValueImputer

class TestMissingValueImputer(unittest.TestCase):

    def setUp(self):
        """
        Prepare a sample dataframe for testing.
        """
        self.df = pd.DataFrame({
            'Age': [23, 25, np.nan, 30, 22],
            'Gender': ['Male', 'Female', 'Female', np.nan, 'Male'],
            'Score': [85, 96, 88, 92, 90],
            'Country': ['US', 'UK', 'CA', np.nan, 'US']
        })
       
        self.imputer = MissingValueImputer(target_column='Score', linear_correlation_threshold=0.8, rf_correlation_threshold=0.3)

    @patch.object(MissingValueImputer, 'impute_using_linear_regression')
    def test_impute_missing_values_linear_regression(self, mock_linear_regression):
        """
        Test to check that missing values are imputed using Linear Regression when correlation is strong.
        """
        self.df['Age'] = self.df['Age'] * 2  # Strong correlation with 'Score'

        imputed_df = self.imputer.impute_missing_values(self.df)
        
        # Verify that the linear regression imputation method was called
        mock_linear_regression.assert_called_once()

        # Ensure that 'Age' is imputed and check type
        self.assertFalse(imputed_df['Age'].isnull().any())
        self.assertIsInstance(imputed_df['Age'].iloc[2], np.float64)
        print("Test passed: Linear Regression used for imputing missing values in 'Age'.")

    @patch.object(MissingValueImputer, 'impute_using_random_forest')
    def test_impute_missing_values_random_forest(self, mock_rf):
        """
        Test to check that missing values are imputed using Random Forest when correlation is moderate.
        """
        self.df['Age'] = self.df['Age'] + np.random.randn(5) * 5  # Moderate correlation with 'Score'

        imputed_df = self.imputer.impute_missing_values(self.df)

        # Verify that the random forest imputation method was called
        mock_rf.assert_called_once()

        # Ensure that 'Age' is imputed and check type
        self.assertFalse(imputed_df['Age'].isnull().any())
        self.assertIsInstance(imputed_df['Age'].iloc[2], np.float64)
        print("Test passed: Random Forest used for imputing missing values in 'Age'.")

    @patch.object(MissingValueImputer, 'impute_using_median')
    def test_impute_missing_values_median(self, mock_median):
        """
        Test to check that missing values are imputed using Median when correlation is weak.
        """
        self.df['Age'] = np.random.randint(20, 30, 5)  # Weak correlation with 'Score'

        imputed_df = self.imputer.impute_missing_values(self.df)

        # Verify that the median imputation method was called
        mock_median.assert_called_once()

        # Check that 'Age' is imputed with the median value
        self.assertFalse(imputed_df['Age'].isnull().any())
        self.assertEqual(imputed_df['Age'].iloc[2], self.df['Age'].median())
        print("Test passed: Median used for imputing missing values in 'Age'.")

    @patch.object(MissingValueImputer, 'impute_using_mode')
    def test_impute_missing_values_mode(self, mock_mode):
        """
        Test to check that missing values in categorical columns are imputed using Mode.
        """
        self.df['Gender'] = self.df['Gender'].fillna('Male')
        self.df['Country'] = self.df['Country'].fillna('US')

        imputed_df = self.imputer.impute_missing_values(self.df)

        # Verify that the mode imputation method was called for categorical columns
        mock_mode.assert_called_once()

        # Check that missing values are filled in 'Gender' and 'Country'
        self.assertFalse(imputed_df['Gender'].isnull().any())
        self.assertFalse(imputed_df['Country'].isnull().any())
        print("Test passed: Mode used for imputing missing values in categorical columns.")

    @patch.object(MissingValueImputer, 'impute_using_median')
    def test_impute_missing_values_no_correlation(self, mock_median):
        """
        Test to check that missing values are imputed with Mode or Median when no correlation exists.
        """
        self.df['Age'] = np.random.randint(20, 50, 5)  # No correlation with 'Score'

        imputed_df = self.imputer.impute_missing_values(self.df)

        # Verify that the median imputation method was called for weak/no correlation
        mock_median.assert_called_once()

        # Ensure 'Age' is imputed and check it's median
        self.assertFalse(imputed_df['Age'].isnull().any())
        self.assertEqual(imputed_df['Age'].iloc[2], self.df['Age'].median())
        print("Test passed: Median used when no strong correlation exists.")

if __name__ == '__main__':
    unittest.main()
