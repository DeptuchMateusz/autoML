import time

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn import tree
import graphviz
import dtreeviz
import os
import pandas as pd
import warnings
import sys
warnings.filterwarnings("ignore", category=UserWarning)


# Suppress output temporarily


def makeplots(best_models, X, y, path):
    #original_stdout = sys.stdout
    #original_stderr = sys.stderr
    #sys.stdout = open(os.devnull, 'w')
    #sys.stderr = open(os.devnull, 'w')

    if not os.path.exists(f"{path}/plots"):
        os.makedirs(f"{path}/plots")

    for model in best_models:

        if model.__class__.__name__ == "DecisionTreeClassifier":
            viz = dtreeviz.model(model, X, y,
                                 target_name="target",
                                 feature_names=X.columns)
            viz.view().save(f"{path}/plots/tree.svg")
        elif model.__class__.__name__ == "LogisticRegression":
            pass #TODO: feature importance?

    for file in os.listdir(f"{path}/results"):
        if file.endswith(".csv"):
            df = pd.read_csv(f"{path}/results/{file}")
            plt.plot(df['f1'])
            plt.plot(df['accuracy'])
            plt.plot(df['precision'])
            plt.plot(df['recall'])
            plt.title(f'{file[:-4]} model')
            plt.ylabel('score')
            plt.xlabel('iteration')
            plt.legend(['f1', 'accuracy', 'precision', 'recall'], loc='upper left')
            plt.savefig(f"{path}/plots/{file}_convergence.png")
            plt.clf()

    #sys.stdout.close()
    #sys.stderr.close()
    #sys.stdout =original_stdout
    #sys.stderr = original_stderr

    return None