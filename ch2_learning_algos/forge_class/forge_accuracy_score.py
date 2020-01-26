from sklearn.model_selection import train_test_split
from mglearn.datasets import make_forge
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

X, y = make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = KNeighborsClassifier(5).fit(X_train, y_train)
preds = clf.predict(X_test)

# Scores 0.86
print("Score {:.2f}".format(clf.score(X_test, y_test)))