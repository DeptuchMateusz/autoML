from medaid import medaid
import pandas as pd
data = pd.read_csv('../data/binary/cardio_train.csv', sep=';')
X = data.drop(columns=['cardio', 'id'])
y = data['cardio']
aid = medaid(X, y, mode="perform", metric="recall", search="random")
aid.train()
print(aid.best_models)