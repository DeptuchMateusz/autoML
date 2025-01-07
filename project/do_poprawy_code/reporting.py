#here will be a class for reporting
#class will be used to generate reports in html

import os
import pandas as pd
from dtreeviz import dtreeviz

from project.do_poprawy_code.medaid import medaid

class Reporting:
    def __init__(self, aid, path):
        self.path = path
        self.aid = aid

    def generate_report(self):
        if not os.path.exists(f"{self.path}/report"):
            os.makedirs(f"{self.path}/report")

        title = "Model comparison report"

        #create empty html file
        f = open(f"{self.path}/report/report.html", "w")
        f.write(f"<html lang='en'>")
        f.write(f"<h1>{title}</h1>")
        f.write("""
            <style>
                .scrollable-container {
                    width: 100%; /* Full width of the page */
                    height: 600px; /* Limit visible height */
                    overflow: auto; /* Enable scrolling */
                    border: 1px solid #ccc; /* Optional border */
                }
                .scrollable-container img {
                    width: 100000px; /* Very large width */
                    height: auto; /* Maintain aspect ratio */
                }
            </style>
            """)
        #analyze the X and y data
        f.write(f"<h2>Data analysis</h2>")
        f.write(f"<p>Number of rows: {len(self.aid.X)}</p>")
        f.write(f"<p>Number of columns: {len(self.aid.X.columns)}</p>")
        #list all columns, their types, number of missing values and  unique values/ranges
        f.write(f"<h3>Columns</h3>")
        f.write(f"<table>")
        f.write(f"<tr>")
        f.write(f"<th>Column</th>")
        f.write(f"<th>Type</th>")
        f.write(f"<th>Missing values</th>")
        f.write(f"<th>Unique values</th>")
        f.write(f"</tr>")
        for column in self.aid.X.columns:
            f.write(f"<tr>")
            f.write(f"<td>{column}</td>")
            #clasisfy the type of the column as binary, categorical or numerical
            f.write(f"<td>{self.aid.X[column].dtype}</td>")
            f.write(f"<td>{self.aid.X[column].isnull().sum()}</td>")
            f.write(f"<td>{self.aid.X[column].nunique()}</td>")
            f.write(f"</tr>")
        f.write(f"</table>")
        #list the distributions of the features
        f.write(f"<h2>Feature distributions</h2>")
        plots_path = os.path.join(self.path, 'medaid', 'distribution_plots')
        print(plots_path)
        for plot_file in os.listdir(plots_path):
            if plot_file.endswith('.png'):
                plot_path = f'../medaid/distribution_plots/{plot_file}'
                print(plot_path)
                #add the plots to the report, five in a row, smaller images
                f.write(f"<img src='{plot_path}' width='250' height='200'>")
        #add correlation matrix and correlation with y plots
        f.write(f"<h2>Correlation matrix</h2>")
        f.write(f"<img src='../medaid/correlation_plots/correlation_matrix.png' width='600' height='600'>")
        f.write(f"<h2>Correlation with y</h2>")
        for plot_file in os.listdir(os.path.join(self.path, 'medaid', 'correlation_plots')):
            if plot_file.endswith('.png'):
                plot_path = f'../medaid/correlation_plots/{plot_file}'
                if plot_path != '../medaid/correlation_plots/correlation_matrix.png':
                    f.write(f"<img src='{plot_path}' width='250' height='200'>")
        #add the head of the data frame
        f.write(f"<h2>Data frame head</h2>")
        f.write(f"<table>")
        f.write(f"<tr>")
        for col in self.aid.X.columns:
            f.write(f"<th>{col}</th>")
        f.write(f"</tr>")
        #add the first 5 rows of the data frame
        for row in self.aid.X.head().values:
            f.write(f"<tr>")
            for value in row:
                #if two unique values, make it a binary value
                if self.aid.X[col].nunique() == 2:
                    f.write(f"<td>{int(value)}</td>")
                else:
                    f.write(f"<td>{value}</td>")
            f.write(f"</tr>")
        f.write(f"</table>")

        #list the target variable
        f.write(f"<p>Target variable: {self.aid.y.name}</p>")
        #check if the target variable is binary
        if self.aid.y.nunique() == 2:
            f.write(f"<p>Binary target variable</p>")
        else:
            f.write(f"<p>Non-binary target variable</p>")
        #chcek if the target variable is balanced
        f.write(f"<p>Target variable balance: {self.aid.y.value_counts().values[0]} negative to  {self.aid.y.value_counts().values[1]} positive </p>")
        #list the models used
        f.write(f"<h2>Models</h2>")
        f.write(f"<p>Models used: {', '.join(self.aid.models)}</p>")
        #list the metric used
        f.write(f"<p>Metric used: {self.aid.metric}</p>")
        f.write(f"<h2>Model ranking</h2>")
        f.write(f"<table>")
        f.write(f"<tr>")
        f.write(f"<th>Model</th>")
        f.write(f"<th>Accuracy</th>")
        f.write(f"<th>Precision</th>")
        f.write(f"<th>Recall</th>")
        f.write(f"<th>F1</th>")
        f.write(f"</tr>")
        for model in self.aid.best_metrics[['model', 'accuracy', 'precision', 'recall', 'f1']].values:
            f.write(f"<tr>")
            f.write(f"<td>{model[0]}</td>")
            f.write(f"<td>{model[1]}</td>")
            f.write(f"<td>{model[2]}</td>")
            f.write(f"<td>{model[3]}</td>")
            f.write(f"<td>{model[4]}</td>")
            f.write(f"</tr>")

        f.write(f"</table>")

        #for each model add the model name, confusion matrix, classification report, feature importance plots and if it is a tree add the tree.svg
        for model in self.aid.best_models:
            f.write(f"<h2>{model.__class__.__name__}</h2>")
            f.write(f"<h3>Confusion matrix</h3>")
            f.write(f"<img src='../medaid/confusion_matrix/{model.__class__.__name__}_confusion_matrix.png' width='400' height='400'>")
            f.write(f"<h3>Feature importance</h3>")
            #display the feature importnace plots
            if os.path.exists(f"{self.path}/medaid/shap_feature_importance/{model.__class__.__name__}_custom_feature_importance.png"):
                f.write(f"<img src='../medaid/shap_feature_importance/{model.__class__.__name__}_custom_feature_importance.png' width='400' height='400'>")
            if model.__class__.__name__ == "DecisionTreeClassifier":
                f.write(f"<h3>Tree</h3>")
                f.write("""
                        <div class="scrollable-container">
                            <img src="../medaid/plots/tree.svg" alt="Decision Tree Visualization">
                        </div>
                        """)
        f.write(f"</html>")

        return None
# main to check if it works
if __name__ == "__main__":
    import pandas as pd
    from project.do_poprawy_code.medaid import medaid
    data = pd.read_csv('../../data/binary/cardio_train.csv', sep=';')
    X = data.drop(columns=['cardio', 'id'])
    y = data['cardio']
    # Create an instance of medaid
    #aid = medaid(X, y, mode="perform", metric="recall", search="random", n_iter=2)
    #aid.train()
    #aid.save()

    #read aid from file
    import pickle
    with open ('../../project/do_poprawy_code/medaid/medaid.pkl', 'rb') as f:
        aid = pickle.load(f)

    path = os.path.dirname(os.path.abspath(__file__))
    report = Reporting(aid, path)
    report.generate_report()
