from naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np # linear algebra


iris = datasets.load_iris()
X = iris.data  # we only take the first 4 features.
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)


NB = GaussianNB()
NB.fit(X_train, y_train)

probs=NB.predict(X_test)
y_pred=np.argmax(probs, 1)


print(f"Accuracy: {sum(y_pred==y_test)/X_test.shape[0]}")