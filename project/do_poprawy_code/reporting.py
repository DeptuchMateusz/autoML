#here will be a class for reporting
#class will be used to generate reports in html

import os
import pandas as pd
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
        # Insert all plots (png) from project/do_poprawy_code/medaid/plots
        f.write(f"<h2>Model convergence</h2>")
        plots_path = os.path.join(self.path, 'medaid', 'plots')
        print(plots_path)
        for plot_file in os.listdir(plots_path):
            if plot_file.endswith('.png'):
                plot_path = f'../medaid/plots/{plot_file}'
                print(plot_path)
                f.write(f"<img src='{plot_path}' alt='{plot_file}'>")
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
