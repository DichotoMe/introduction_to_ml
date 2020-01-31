from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

from mglearn.datasets import load_extended_boston

X, y = load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Training score: 0.95, test score: 0.61
lr = LinearRegression().fit(X_train, y_train)
print("Training score: {:.2f}, test score: {:.2f}".format(lr.score(X_train, y_train),
                                                          lr.score(X_test, y_test)))

# Training score: 0.93, test score: 0.77
ridge = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training score: {:.2f}, test score: {:.2f}".format(ridge.score(X_train, y_train),
                                                          ridge.score(X_test, y_test)))

# Training score: 0.90, test score: 0.77
lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training score: {:.2f}, test score: {:.2f}".format(lasso.score(X_train, y_train),
                                                          lasso.score(X_test, y_test)))