import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from mglearn.datasets import make_wave
import matplotlib.pyplot as plt

X, y = make_wave()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

fig, axes = plt.subplots(1, 3, figsize=(30, 8))

line = np.linspace(-3, 3, 1000).reshape(-1, 1)

for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors).fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, 'k^', markersize=5)
    ax.plot(X_test, y_test, 'rv', markersize=5)

    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score {:.2f}"
                 .format(n_neighbors,
                         reg.score(X_train, y_train),
                         reg.score(X_test, y_test)))

    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
    axes[0].legend(["Model predictions", "Training data/target", "Test data/target"],
                   loc='best')

plt.show()