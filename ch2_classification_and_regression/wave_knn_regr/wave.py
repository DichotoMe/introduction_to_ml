import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsRegressor

import mglearn
from mglearn import cm3
from mglearn.datasets import make_wave

X, y = make_wave(n_samples=40)
X_test = np.array([[-1.5], [0.9], [1.5]])


def plot_knn_regression(X_train, y_train, X_test, n_neighbors=1):
    dists = skl.metrics.euclidean_distances(X_test, X_train)
    closest = np.argsort(dists, axis=1)

    plt.figure(figsize=(10, 6))

    reg = KNeighborsRegressor(n_neighbors).fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    for x, y, neighbors in zip(X_test, y_pred, closest):
        for neighbor in neighbors[:n_neighbors]:
            plt.arrow(x[0], y,
                      X_train[neighbor, 0] - x[0], y_train[neighbor] - y,
                      head_width=0, fc='k', ec='k', alpha=0.03)

    train, = plt.plot(X_train, y_train, 'co')
    test, = plt.plot(X_test, -3 * np.ones(len(X_test)), 'k^', markersize=10)
    pred, = plt.plot(X_test, y_pred, 'bp', markersize=10)
    plt.vlines(X_test, -3.1, 3.1, linestyle="--")
    plt.legend([train, test, pred],
               ["training data/tagret", "test data", "test prediction"],
               ncol=3,
               loc=(0.1, 1.025))
    plt.ylim(-3.1, 3.1)
    plt.xlabel("Feature")
    plt.ylabel("Target")


plot_knn_regression(X, y, X_test, n_neighbors=9)

plt.show()