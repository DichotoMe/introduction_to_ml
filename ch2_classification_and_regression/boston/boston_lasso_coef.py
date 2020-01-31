from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mglearn.datasets import load_extended_boston
import numpy as np

X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)
r1 = Lasso(alpha=1, max_iter=10000).fit(X_train, y_train)
r001 = Ridge(alpha=0.01, max_iter=10000).fit(X_train, y_train)
r00001 = Ridge(alpha=0.0001, max_iter=10000).fit(X_train, y_train)

plt.title("Lasso. {:} features".format(X.shape[1]))
plt.plot(r1.coef_, 's', label="Alpha = 1, Features: {}".format(np.sum(r1.coef_ != 0)))
plt.plot(r001.coef_, '^', label="Alpha = 0.01, Features: {}".format(np.sum(r001.coef_ != 0)))
plt.plot(r00001.coef_, 'v', label="Alpha = 0.0001, Features: {}".format(np.sum(r00001.coef_ != 0)))

plt.plot(lr.coef_, 'o', label="Least squares")

plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend(ncol=2)

plt.show()