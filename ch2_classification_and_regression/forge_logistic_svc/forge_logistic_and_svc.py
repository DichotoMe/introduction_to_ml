import sklearn as skl
import mglearn.plots as mgplt
from mglearn.datasets import make_forge
import mglearn
import matplotlib.pyplot as plt

X, y = make_forge()

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

mglearn.plots.plot_linear_svc_regularization()

for model, ax in zip([skl.linear_model.LogisticRegression(), skl.svm.SVC()], axes):
    clf = model.fit(X, y)
    mgplt.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=0.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

axes[0].legend()
plt.show()