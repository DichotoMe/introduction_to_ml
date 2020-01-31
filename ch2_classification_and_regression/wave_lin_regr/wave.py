import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

from mglearn.datasets import make_wave

X, y = make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
line = np.linspace(-3, 3, 100).reshape(-1, 1)
lr = LinearRegression().fit(X_train, y_train)
print(lr.coef_[0], lr.intercept_)

plt.figure(figsize=(8, 8))
plt.plot(line, lr.predict(line))
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['left'].set_position('center')
ax.set_ylim(-3, 3)
ax.set_title("Training score {:.2f}, test score {:.2f}. \nLeast squares perform OK on small data".format(lr.score(X_train, y_train),
                                                               lr.score(X_test, y_test)))
ax.legend(['model', 'training data'], loc='best')
ax.grid(True)
ax.set_aspect('equal')

plt.scatter(X_train, y_train)

plt.show()
