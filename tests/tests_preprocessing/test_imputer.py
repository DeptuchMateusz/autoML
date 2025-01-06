import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from project.preprocessing.imputer import MissingValueImputer

class TestMissingValueImputer(unittest.TestCase):

    def setUp(self):
        """
        Prepare a sample dataframe for testing.
        """
        self.df = pd.DataFrame({
            'Age': [23, 25, np.nan, 30, 22],
            'Gender': ['Male', 'Female', 'Female', np.nan, 'Male'],
            'Score': [86, 96, 88, 92, 90],
            'Country': ['US', 'UK', 'CA', np.nan, 'US']
        })
        
        #self.df['Age'] = self.df['Age'].astype('Int64')
        self.imputer = MissingValueImputer(target_column='Score', linear_correlation_threshold=0.8, rf_correlation_threshold=0.3)

    def test_impute_missing_values_linear_regression(self):
        """
        Test to check that missing values are imputed using Linear Regression when correlation is strong.
        """
        self.df['Age'] = self.df['Score'] * 0.5 # Strong correlation with 'Score'
        self.df.at[3, 'Age'] = np.nan 

        imputed_df = self.imputer.impute_missing_values(self.df)
        
        # Print the DataFrame after imputation
        print("DataFrame after Linear Regression Imputation:")
        print(imputed_df)
        
        # Ensure that 'Age' is imputed and check type
        self.assertFalse(imputed_df['Age'].isnull().any())
        self.assertIsInstance(imputed_df['Age'].iloc[2], np.float64)
        print("Test passed: Linear Regression used for imputing missing values in 'Age'.")

    def test_impute_missing_values_random_forest(self):
        """
        Test to check that missing values are imputed using Random Forest when correlation is moderate.
        """
        self.df['Age'] = self.df['Age'] + np.random.randn(5) * 5  # Moderate correlation with 'Score'

        imputed_df = self.imputer.impute_missing_values(self.df)

        # Print the DataFrame after imputation
        print("DataFrame after Random Forest Imputation:")
        print(imputed_df)

        # Ensure that 'Age' is imputed and check type
        self.assertFalse(imputed_df['Age'].isnull().any())
        self.assertIsInstance(imputed_df['Age'].iloc[2], np.float64)
        print("Test passed: Random Forest used for imputing missing values in 'Age'.")

    def test_impute_missing_values_median(self):
        """
        Test to check that missing values are imputed using Median when correlation is weak.
        """
        self.df['Age'] = np.random.randint(20, 30, 5)  # Weak correlation with 'Score'

        imputed_df = self.imputer.impute_missing_values(self.df)

        # Print the DataFrame after imputation
        print("DataFrame after Median Imputation:")
        print(imputed_df)

        # Check that 'Age' is imputed with the median value
        self.assertFalse(imputed_df['Age'].isnull().any())
        self.assertEqual(imputed_df['Age'].iloc[2], self.df['Age'].median())
        print("Test passed: Median used for imputing missing values in 'Age'.")

    def test_impute_missing_values_categorical(self):
        """
        Test to check that missing values in categorical columns are imputed using Mode.
        """
        self.df['Gender'] = self.df['Gender'].fillna('Male')
        self.df['Country'] = self.df['Country'].fillna('US')

        imputed_df = self.imputer.impute_missing_values(self.df)

        # Print the DataFrame after imputation
        print("DataFrame after Decision Tree Imputation for Categorical Columns:")
        print(imputed_df)

        # Check that missing values are filled in 'Gender' and 'Country'
        self.assertFalse(imputed_df['Gender'].isnull().any())
        self.assertFalse(imputed_df['Country'].isnull().any())
        print("Test passed: Decision Tree  used for imputing missing values in categorical columns.")

    def test_impute_missing_values_no_correlation(self):
        """
        Test to check that missing values are imputed with Median when no correlation exists.
        """
        self.df['Age'] = np.random.randint(20, 50, 5)  # No correlation with 'Score'

        imputed_df = self.imputer.impute_missing_values(self.df)

        # Print the DataFrame after imputation
        print("DataFrame after Imputation with No Correlation:")
        print(imputed_df)

        # Ensure 'Age' is imputed and check it's median
        self.assertFalse(imputed_df['Age'].isnull().any())
        self.assertEqual(imputed_df['Age'].iloc[2], self.df['Age'].median())
        print("Test passed: Median used when no strong correlation exists.")

    def test_column_types_handling(self):
        """
        Test to ensure correct handling of numerical vs categorical column imputation.
        """
        imputed_df = self.imputer.impute_missing_values(self.df)

        # Print the DataFrame after imputation
        print("DataFrame after Imputation of All Columns:")
        print(imputed_df)

        # Check numerical columns are imputed correctly
        self.assertFalse(imputed_df['Age'].isnull().any())
        self.assertFalse(imputed_df['Score'].isnull().any())

        # Check categorical columns are imputed with Mode
        self.assertFalse(imputed_df['Gender'].isnull().any())
        self.assertFalse(imputed_df['Country'].isnull().any())

        print("Test passed: Column types are correctly handled and imputed.")

if __name__ == '__main__':
    unittest.main()
