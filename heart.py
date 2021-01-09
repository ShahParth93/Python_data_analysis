import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier


import joblib

data = pd.read_csv("./heart.csv")

data = data[["thal","cp","thalach","ca","oldpeak","target"]]

X = data.drop(['target'],axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1951)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
print(acc_random_forest)

joblib.dump(random_forest,"./model_saved11")

