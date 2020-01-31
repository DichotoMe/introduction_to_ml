from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mglearn.datasets import load_extended_boston

X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)
r1 = Ridge(alpha=1).fit(X_train, y_train)
r01 = Ridge(alpha=0.1).fit(X_train, y_train)
r10 = Ridge(alpha=10).fit(X_train, y_train)

plt.title("{:} features".format(X.shape[1]))
plt.plot(r1.coef_, 's', label="Ridge alpha = 1")
plt.plot(r01.coef_, '^', label="Ridge alpha = 0.1")
plt.plot(r10.coef_, 'v', label="Ridge alpha = 10")

plt.plot(lr.coef_, 'o', label="Least squares")

plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

plt.show()