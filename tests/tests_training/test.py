from medaid.training.medaid import medaid


medaid = medaid(dataset_path='../../data/binary/cardio_train.csv', target_column='cardio', metric="recall", search="random", n_iter=2)
medaid.train()
assert medaid.models_ranking() == ['LogisticRegression', 'RandomForestClassifier'], "Models ranking is not as expected"
medaid.report()


