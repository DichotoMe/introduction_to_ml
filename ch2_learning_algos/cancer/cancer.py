from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

from util.util import plot_training_vs_test_accuracy_by_neighbors

# Measure training and test accuracy
# for breast_cancer dataset
# using K-neighbor classifier
# with K from 1 through 10
cancer = load_breast_cancer()

plot_training_vs_test_accuracy_by_neighbors(cancer.data,
                                            cancer.target,
                                            method_class=KNeighborsClassifier,
                                            neighbors_range=range(1, 11),
                                            random_state=66,
                                            stratify=cancer.target)