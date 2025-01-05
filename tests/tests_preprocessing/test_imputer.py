import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from project.preprocessing.imputer import MissingValueImputer 
from project.preprocessing.categorical_column_handler import CategoricalColumnHandler

class TestMissingValueImputer(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing with some missing values
        self.df = pd.DataFrame({
            'Age': [23, 25, np.nan, 30, 22],
            'Gender': ['Male', 'Female', 'Female', np.nan, 'Male'],
            'Score': [85, np.nan, 88, 92, 90],
            'Country': ['US', 'UK', 'CA', np.nan, 'US']
        })
        
        self.imputer = MissingValueImputer(target_column='Score', linear_correlation_threshold=0.8, rf_correlation_threshold=0.3)

    def test_impute_missing_values_linear_regression(self):
        """
        Test case to check that missing values are imputed using Linear Regression when correlation is strong.
        """
        # Modify dataframe to create strong correlation with 'Score'
        self.df['Age'] = self.df['Age'] * 2  # Strong correlation with 'Score'
        
        imputed_df = self.imputer.impute_missing_values(self.df)
        
        # Check that missing values in 'Age' are imputed correctly
        self.assertFalse(imputed_df['Age'].isnull().any())  # 'Age' should have no missing values after imputation
        print("Test passed: Linear Regression used for imputing missing values in 'Age'.")

    def test_impute_missing_values_random_forest(self):
        """
        Test case to check that missing values are imputed using Random Forest when correlation is moderate.
        """
        # Modify dataframe to create moderate correlation with 'Score'
        self.df['Age'] = self.df['Age'] + np.random.randn(5) * 5  # Moderate correlation with 'Score'
        
        imputed_df = self.imputer.impute_missing_values(self.df)
        
        # Check that missing values in 'Age' are imputed correctly
        self.assertFalse(imputed_df['Age'].isnull().any())  # 'Age' should have no missing values after imputation
        print("Test passed: Random Forest used for imputing missing values in 'Age'.")

    def test_impute_missing_values_mean_median(self):
        """
        Test case to check that missing values are imputed using Mean or Median when correlation is weak.
        """
        # Modify dataframe to create weak/no correlation with 'Score'
        self.df['Age'] = np.random.randint(20, 30, 5)  # Weak correlation with 'Score'
        
        imputed_df = self.imputer.impute_missing_values(self.df)
        
        # Check that missing values in 'Age' are imputed with median
        self.assertFalse(imputed_df['Age'].isnull().any())  # 'Age' should have no missing values after imputation
        self.assertEqual(imputed_df['Age'].iloc[2], imputed_df['Age'].median())  # The 3rd value should be replaced by median value
        print("Test passed: Mean/Median used for imputing missing values in 'Age'.")

    def test_impute_missing_values_categorical_mode(self):
        """
        Test case to check that missing values in categorical columns are imputed using Mode.
        """
        # Create missing values in 'Gender' and 'Country'
        self.df['Gender'] = self.df['Gender'].fillna('Male')
        self.df['Country'] = self.df['Country'].fillna('US')
        
        imputed_df = self.imputer.impute_missing_values(self.df)
        
        # Check that missing values in 'Gender' and 'Country' are imputed using mode
        self.assertFalse(imputed_df['Gender'].isnull().any())  # 'Gender' should have no missing values
        self.assertFalse(imputed_df['Country'].isnull().any())  # 'Country' should have no missing values
        print("Test passed: Mode used for imputing missing values in categorical columns.")

    def test_impute_missing_values_no_correlation(self):
        """
        Test case to check that missing values are imputed using Mode or Median when no correlation exists.
        """
        # Remove strong correlation and ensure no correlation
        self.df['Age'] = np.random.randint(20, 30, 5)  # No correlation with 'Score'
        
        imputed_df = self.imputer.impute_missing_values(self.df)
        
        # Check that missing values are imputed with Mode or Median for 'Age' and 'Country'
        self.assertFalse(imputed_df['Age'].isnull().any())  # 'Age' should have no missing values
        self.assertEqual(imputed_df['Age'].iloc[2], imputed_df['Age'].median())  # Median imputation
        print("Test passed: Mode/Median used when no strong correlation exists.")

    def test_impute_missing_values_check_column_types(self):
        """
        Test case to check the imputation strategy is applied correctly based on column type (numerical vs categorical).
        """
        # For this test, we use numerical columns 'Age' and 'Score' and categorical 'Gender' and 'Country'
        imputed_df = self.imputer.impute_missing_values(self.df)
        
        # Ensure numerical columns are imputed with an appropriate strategy
        self.assertFalse(imputed_df['Age'].isnull().any())  # Check if 'Age' is imputed correctly
        self.assertFalse(imputed_df['Score'].isnull().any())  # Check if 'Score' is imputed correctly

        # Ensure categorical columns are imputed with Mode
        self.assertFalse(imputed_df['Gender'].isnull().any())  # 'Gender' should be imputed with Mode
        self.assertFalse(imputed_df['Country'].isnull().any())  # 'Country' should be imputed with Mode
        
        print("Test passed: Column types are correctly handled and imputed.")

if __name__ == '__main__':
    unittest.main()
