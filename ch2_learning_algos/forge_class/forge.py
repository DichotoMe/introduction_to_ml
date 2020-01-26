import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

import mglearn.datasets
# generate forge dataset
X, y = mglearn.datasets.make_forge()


def plot_knn_classification(X_train, y_train, X_test, n_neighbors=1):
    dists = euclidean_distances(X_train, X_test)
    closest_idx = np.argsort(dists, axis=0)

    """for x, neigbors in zip(X_test, closest_idx.T):
        for nb in neigbors[:n_neighbors]:
            plt.arrow(
                x[0], x[1],
                X_train[nb, 0] - x[0], X_train[nb, 1] - x[1],
                head_width=0, fc='k', ec='k'
            )"""

    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    test_pts = mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], clf.predict(X_test), markers=["*", "p"])
    training_pts = mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.legend(
        training_pts + test_pts,
        ["training class 0", "training class 1", "test pred 0", "test pred 1", "test pred 2"]
    )


X_test = np.array([[8.2, 3.66214339], [9.9, 3.2], [11.2, .5]])

# one-neighbor classification
plot_knn_classification(X, y, X_test)
# k-neighbours classification
plot_knn_classification(X, y, X_test, n_neighbors=5)

plt.show()
