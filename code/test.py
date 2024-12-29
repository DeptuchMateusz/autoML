from medaid import medaid
import pandas as pd
data = pd.read_csv('/Users/mateuszdeptuch/SCHOOL/AUTOML/projekt2/data/binary/cardio_train.csv', sep=';')
X = data.drop(columns=['cardio', 'id'])
y = data['cardio']
aid = medaid(X, y)
aid.train()
print(aid.best_models)
print(aid.best_models_scores)