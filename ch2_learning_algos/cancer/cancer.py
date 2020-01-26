from sklearn.datasets import load_breast_cancer
import numpy as np

d = {'adfgfdswdfg': 1}
cancer = load_breast_cancer()
print(dict(zip(cancer.target_names, np.bincount(cancer.target))))
print(cancer.feature_names)