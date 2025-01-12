import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np

class PredictExplainer:
    def __init__(self, medaid, model):
        """
        Initializes the PredictExplainer class with the medaid object and the model.

        Parameters:
        - medaid: The medaid object that holds the necessary data or configurations.
        - model: The model used for making predictions.
        """
        self.medaid = medaid
        self.model = model
        self.preprocessing_details = pd.read_csv(self.medaid.path + '/results/preprocessing_details.csv')

        # Create dictionaries for encoders, scalers, and imputers
        self.encoders = {}
        self.scalers = {}
        self.imputers = {}

    def preprocess_input_data(self, input_data):
        """
        Preprocesses the input data using the stored preprocessing details.
        This version uses pandas get_dummies for one-hot encoding.
        """
        processed_data = input_data.copy()

        # Handle one-hot encoding using pandas get_dummies
        for _, row in self.preprocessing_details.iterrows():
            column_name = row["Column Name"]

            if column_name not in processed_data.columns:
                continue  # Skip columns not in the input data

            # Encoding handling: Apply encoding methods first
            if row["Encoded"] and not row["Removed"]:
                encoding_method = row["Encoding Method"]
                if encoding_method == "One-Hot Encoding":
                    # One-Hot encoding using pandas get_dummies
                    # For input_data, apply get_dummies and align columns
                    processed_data_encoded = pd.get_dummies(processed_data, columns=[column_name], drop_first=False)
                    # Ensure that columns in the input data match with the training data
                    processed_data_encoded = processed_data_encoded.reindex(columns=self.medaid.X.columns, fill_value=0)
                    processed_data = processed_data_encoded

                elif encoding_method == "Label Encoding":
                    if column_name not in self.encoders:
                        label_encoder = LabelEncoder()
                        if column_name in self.medaid.X.columns:
                            label_encoder.fit(self.medaid.X[column_name])
                        else:
                            label_encoder.fit(processed_data[column_name])
                        self.encoders[column_name] = label_encoder

                    # Transform the input data
                    processed_data[column_name] = self.encoders[column_name].transform(processed_data[column_name])

        # Now, handle imputation after encoding
        for _, row in self.preprocessing_details.iterrows():
            column_name = row["Column Name"]

            if column_name not in processed_data.columns:
                continue  # Skip columns not in the input data

            # Imputation handling
            if row["Imputation Method"] and not row["Removed"]:
                imputation_method = row["Imputation Method"]
                strategy = imputation_method.lower() if pd.notna(imputation_method) else "mean"

                if column_name not in self.imputers:
                    if processed_data[column_name].dtype in [np.float64, np.int64]:  # Numeric data
                        if strategy in ["mean", "median", "most_frequent"]:
                            imputer = SimpleImputer(strategy=strategy)
                        else:
                            raise ValueError(f"Unsupported imputation strategy for numeric data: {strategy}")
                    else:  # Categorical data
                        if strategy == "mean":
                            strategy = "most_frequent"  # Automatically change "mean" to "most_frequent" for categorical data
                        if strategy == "most_frequent":
                            imputer = SimpleImputer(strategy="most_frequent")
                        else:
                            raise ValueError(f"Unsupported imputation strategy for categorical data: {strategy}")

                    # Fit the imputer
                    if column_name in self.medaid.X.columns:
                        imputer.fit(self.medaid.X[[column_name]])
                    else:
                        imputer.fit(processed_data[[column_name]])

                    # Save the fitted imputer
                    self.imputers[column_name] = imputer

                # Transform the input data
                processed_data[column_name] = self.imputers[column_name].transform(
                    processed_data[column_name].values.reshape(-1, 1)
                ).flatten()

        # Scaling handling: Scale after encoding and imputation
        for _, row in self.preprocessing_details.iterrows():
            column_name = row["Column Name"]

            if column_name not in processed_data.columns:
                continue  # Skip columns not in the input data

            # Scaling handling
            if row["Scaling Method"] and not row["Removed"]:
                scaling_method = row["Scaling Method"].lower()

                if column_name not in self.scalers:
                    if scaling_method in ["standard scaling", "standardization"]:
                        scaler = StandardScaler()
                    elif scaling_method in ["min-max scaling", "normalization", "min_max"]:
                        scaler = MinMaxScaler()
                    else:
                        raise ValueError(f"Unsupported scaling method: {scaling_method}")

                    if column_name in self.medaid.X.columns:
                        scaler.fit(self.medaid.X[[column_name]])
                    else:
                        scaler.fit(processed_data[[column_name]])

                    self.scalers[column_name] = scaler

                # Transform the input data
                processed_data[column_name] = self.scalers[column_name].transform(
                    processed_data[column_name].values.reshape(-1, 1)
                ).flatten()

        return processed_data

    def predict_target(self, input_data):
        """
        Preprocesses the input data and predicts the target feature using the model.
        """
        # Preprocess the input data first
        processed_input_data = self.preprocess_input_data(input_data)

        # Make the prediction
        # if xgboost enable categorical features
        if self.model.__class__.__name__ == 'XGBClassifier':
            for column in self.medaid.X.columns:
                if self.medaid.X[column].dtype == 'object':
                    processed_input_data[column] = processed_input_data[column].astype('category')

        prediction = self.model.predict(processed_input_data)[0]  # Assuming single prediction for a single patient

        # Get target column name
        target_column = self.medaid.target_column

        # Perform prediction analysis
        prediction_analysis = self.analyze_prediction(prediction, target_column)

        return prediction, prediction_analysis

    def analyze_prediction(self, prediction, target_column):
        """
        Analyzes the predicted value of the target feature and compares it to the dataset.
        """
        # Get target data
        df = self.medaid.df_before

        target_values = df[target_column]

        # Basic comparison with the classes distribution
        value_counts = target_values.value_counts(normalize=True) * 100

        if len(value_counts) == 2:  # Binary classification
            analysis = f"""
            <div class="feature">
                <div class="feature-header" onclick="toggleFeature('target') >Target Feature: {target_column} - Predicted Value: {self._format_value(prediction)}</div>
                <div class="feature-content" id="target">
                    The predicted value for the target feature is <span>{self._format_value(prediction)}</span>.
                    <div class="feature-category">
                        <strong>Prediction Analysis (Binary Classification):</strong>
                        <ul>
                            <li>Class 0 occurs in {value_counts.get(0, 0):.2f}% of patients.</li>
                            <li>Class 1 occurs in {value_counts.get(1, 0):.2f}% of patients.</li>
                            <li>The predicted class of {self._format_value(prediction)} is {'common' if value_counts.get(prediction, 0) > 50 else 'rare'} in the dataset.</li>
                        </ul>
                    </div>
                </div>
            </div>
            """
        else:  # Multiclass classification
            analysis = f"""
            <div class="feature">
                <div class="feature-header" onclick="toggleFeature('target') >Target Feature: {target_column} - Predicted Value: {self._format_value(prediction)}</div>
                <div class="feature-content" id="target">
                    The predicted value for the target feature is <span>{self._format_value(prediction)}</span>.
                    <div class="feature-category">
                        <strong>Prediction Analysis (Multiclass Classification):</strong>
                        <ul>
                            <li>Class distribution:</li>
                            <ul>
            """
            for class_label, percentage in value_counts.items():
                analysis += f"<li>Class {class_label}: {percentage:.2f}% of patients.</li>"

            analysis += f"""
                            </ul>
                            <li>The predicted class of {self._format_value(prediction)} is {'common' if value_counts.get(prediction, 0) > 50 else 'rare'} in the dataset.</li>
                        </ul>
                    </div>
                </div>
            </div>
            """

        return analysis

    def _format_value(self, value):
        """
        Helper function to format values as integers or floating-point numbers with 3 decimal places.
        """
        if isinstance(value, float):
            return f"{value:.3f}"
        elif isinstance(value, int):
            return f"{value}"
        return value

    def generate_html_report(self, df, input_data):
        """
        Generates an HTML report that compares the input data with the dataset.
        """
        # Classify and analyze features
        feature_analysis = self.classify_and_analyze_features(df, input_data)

        # Predict and analyze the target
        prediction, prediction_analysis = self.predict_target(input_data)

        # Start HTML report
        html_report = f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background-color: #f4f4f4;
                        color: #333;
                    }}
                    .container {{
                        width: 80%;
                        margin: 0 auto;
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
                    }}

                    .feature {{
                        margin-bottom: 20px;
                    }}
                    .feature-header {{
                        background-color: #003366;
                        color: white;
                        padding: 10px;
                        border-radius: 5px;
                        font-weight: bold;
                        cursor: pointer;
                    }}
                    .feature-content {{
                        display: none;
                        background-color: #f9f9f9;
                        padding: 10px;
                        border-left: 4px solid #003366;
                        border-radius: 5px;
                        margin-top: 10px;
                    }}
                    .feature-category {{
                        margin-top: 10px;
                    }}
                    .feature-value {{
                        font-weight: bold;
                        color: #003366;
                    }}
                    .btn {{
                        padding: 10px 20px;
                        background-color: #003366;
                        color: white;
                        border: none;
                        cursor: pointer;
                    }}
                    .btn:hover {{
                        background-color: #00509e;
                    }}
                    .prediction {{
                        margin-top: 30px;
                        background-color: #E1E9F1;
                        padding: 20px;
                        border-radius: 8px;
                    }}
                    .prediction-header {{
                        font-weight: bold;
                        color: #003366;
                        margin-bottom: 10px;
                    }}
                    .prediction-content {{
                        color: #333;
                    }}
                    .features-container {{
                        max-height: 500px;
                        overflow-y: scroll;
                        padding-right: 15px;
                    }}
                </style>
                <script>
                    function toggleFeature(id) {{
                        var content = document.getElementById(id);
                        if (content.style.display === "none") {{
                            content.style.display = "block";
                        }} else {{
                            content.style.display = "none";
                        }}
                    }}
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>Patient Data and Prediction Report</h1>

                    <!-- Prediction Section -->
                    <div class="prediction">
                        <div class="prediction-header">Prediction</div>
                        <div class="prediction-content">
                            <p>{prediction_analysis}</p>
                        </div>
                    </div>

                    <!-- Feature Analysis Section -->
                    {feature_analysis}
                </div>
            </body>
        </html>
        """
        return html_report

    def classify_and_analyze_features(self, df, input_data):
        """
        Classifies features into categories (binary, categorical_strings, categorical_numbers, numerical_continuous)
        and analyzes each one based on its type.
        """
        feature_analysis = ""  # Holds the HTML for all features

        for column in df.columns:
            # Initialize the content for this specific feature
            feature_content = ""

            # Check if the column is categorical (i.e., contains strings)
            if df[column].dtype == 'object':
                feature_content = self._analyze_categorical_strings(df, column, input_data[column])
            else:
                # Determine if the column is binary, categorical numeric, or continuous numeric
                unique_values = df[column].nunique()

                if unique_values == 2:  # Binary feature (usually encoded as 0 and 1)
                    feature_content = self._analyze_binary(df, column, input_data[column])
                elif 2 < unique_values < 15 and df[column].dtype in ['int64', 'float64']:  # Categorical numbers
                    feature_content = self._analyze_categorical_numbers(df, column, input_data[column])
                elif df[column].dtype in ['int64', 'float64']:  # Numerical continuous (e.g., age, BMI)
                    feature_content = self._analyze_numerical_continuous(df, column, input_data[column])

            # Add the individual feature analysis wrapped in a <div class="feature"> tag
            feature_analysis += f"<div class='feature'>{feature_content}</div>"

        # Only wrap the entire features inside one .features-container
        return f"<div class='features-container'>{feature_analysis}</div>"

    def _analyze_binary(self, df, column, input_value):
        """
        Generates HTML for a binary feature with collapsible content.
        """
        input_value = input_value.iloc[0]  # Ensure scalar
        value_counts = df[column].value_counts(normalize=True) * 100
        return f"""
            <div class="feature-header" onclick="toggleFeature('{column}_binary')">Feature '{column}' - Value: {input_value}</div>
            <div class="feature-content" id="{column}_binary">
                The new patient has a value of <span class="feature-value">{input_value}</span>.
                <div class="feature-category">This value occurs in <span class="feature-value">{value_counts[input_value]:.3f}%</span> of other patients.</div>
            </div>
        """

    def _analyze_categorical_numbers(self, df, column, input_value):
        """
        Generates HTML for a categorical numeric feature with collapsible content.
        """
        input_value = input_value.iloc[0]  # Ensure scalar
        value_counts = df[column].value_counts(normalize=True) * 100
        categories = [f"Value '{value}' occurs in {count:.3f}% of patients." for value, count in value_counts.items()]

        return f"""
            <div class="feature-header" onclick="toggleFeature('{column}_categorical_numbers')">Feature '{column}' - Value: {input_value}</div>
            <div class="feature-content" id="{column}_categorical_numbers">
                The new patient has a value of <span class="feature-value">{input_value}</span>.
                <div class="feature-category">
                    {'This value is rare in the dataset.' if input_value not in value_counts.index else f'This value occurs in <span class="feature-value">{value_counts[input_value]:.3f}%</span> of patients.'}
                </div>
                <div class="feature-list">
                    <strong>All possible categories and their frequencies:</strong>
                    <ul>
                        {''.join([f"<li>{cat}</li>" for cat in categories])}
                    </ul>
                </div>
            </div>
        """

    def _analyze_categorical_strings(self, df, column, input_value):
        """
        Generates HTML for a categorical string feature with collapsible content.
        """
        input_value = input_value.iloc[0]  # Ensure scalar
        value_counts = df[column].value_counts(normalize=True) * 100

        return f"""
            <div class="feature-header" onclick="toggleFeature('{column}_categorical_strings')">Feature '{column}' - Value: '{input_value}'</div>
            <div class="feature-content" id="'{column}_categorical_strings'">
                The new patient has a value of <span class="feature-value">'{input_value}'</span>.
                <div class="feature-category">
                    {'This value is rare in the dataset.' if input_value not in value_counts.index else f'This value occurs in <span class="feature-value">{value_counts[input_value]:.3f}%</span> of patients.'}
                </div>
            </div>
        """

    def _analyze_numerical_continuous(self, df, column, input_value):
        """
        Generates HTML for a continuous numerical feature with collapsible content.
        """
        mean = df[column].mean()
        median = df[column].median()
        std_dev = df[column].std()
        min_value = df[column].min()
        max_value = df[column].max()

        input_value = input_value.iloc[0]  # Ensure scalar
        if input_value > mean + std_dev:  # Way above
            comparison = "way above"
        elif input_value > mean:  # Slightly above
            comparison = "slightly above"
        elif input_value == mean:  # Equal to mean
            comparison = "equal to"
        elif input_value < mean - std_dev:  # Way below
            comparison = "way below"
        else:  # Slightly below
            comparison = "slightly below"

        return f"""
            <div class="feature-header" onclick="toggleFeature('{column}_numerical_continuous')">Feature '{column}' - Value: {input_value:.3f}</div>
            <div class="feature-content" id="{column}_numerical_continuous">
                The new patient has a value of <span class="feature-value">{input_value:.3f}</span>. This value is {comparison} the mean value of <span class="feature-value">{mean:.3f}</span> for the dataset.
                <div class="feature-category">
                    <strong>Additional details:</strong>
                    <ul>
                        <li>Median value: {median:.3f}</li>
                        <li>Standard deviation: {std_dev:.3f}</li>
                        <li>Min: {min_value:.3f}</li>
                        <li>Max: {max_value:.3f}</li>
                    </ul>
                </div>
            </div>
        """


# Example Usage
if __name__ == "__main__":
    # Load the medaid object
    with open('medaid1/medaid.pkl', 'rb') as file:
        medaid = pickle.load(file)

    model = medaid.best_models[0]
    pe = PredictExplainer(medaid, model)

    # Prepare the input data
    df = medaid.df_before.drop(columns=[medaid.target_column])
    input_data = medaid.df_before.head(1).drop(columns=[medaid.target_column])

    # Generate the HTML report
    html_report = pe.generate_html_report(df, input_data)
    with open('report_predict_and_features.html', 'w') as f:
        f.write(html_report)
