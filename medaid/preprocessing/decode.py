import pandas as pd
import numpy as np

def decode(y, target_column, path):

    labels = pd.read_csv(path + "/results/preprocessing_details.csv")
    labels = labels[labels["Column Name"] == target_column].loc[:, "Label Encoding Mapping"].values[0]
    if not labels:
        return y
    labels = eval(labels)
    labels = {v: k for k, v in labels.items()}
    return [labels[i] for i in y]

