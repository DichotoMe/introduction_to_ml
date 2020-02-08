import sklearn.datasets
from sklearn import linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=202)

# Training set score: 0.92 for C = 0.001
# Test set score: 0.92 for C = 0.001

# Training set score: 0.96 for C = 1
# Test set score: 0.94 for C = 1

# Training set score: 0.97 for C = 100
# Test set score: 0.96 for C = 100

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = linear_model.LogisticRegression(C=C, penalty='l2', solver='liblinear').fit(X_train, y_train)
    print("Training set score: {:.2f} for C = {}".format(lr_l1.score(X_train, y_train), C))
    print("Test set score: {:.2f} for C = {}".format(lr_l1.score(X_test, y_test), C))
    plt.plot(lr_l1.coef_.T, marker, label="C={}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.show()