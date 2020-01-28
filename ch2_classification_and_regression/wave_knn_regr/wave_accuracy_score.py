from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from mglearn.datasets import make_wave

X, y = make_wave(n_samples=40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(3).fit(X_train, y_train)

# Scores 0.83
print("Score R^2 {:.2f}".format(reg.score(X_test, y_test)))
