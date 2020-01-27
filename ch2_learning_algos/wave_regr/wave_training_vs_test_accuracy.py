from sklearn.neighbors import KNeighborsRegressor
from mglearn.datasets import make_wave
from util.util import plot_training_vs_test_accuracy_by_neighbors

X, y = make_wave()

plot_training_vs_test_accuracy_by_neighbors(X, y,
                                            method_class=KNeighborsRegressor,
                                            neighbors_range=range(1, 21),
                                            stratify=None,
                                            random_state=0)
