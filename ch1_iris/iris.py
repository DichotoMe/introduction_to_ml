from matplotlib import pyplot, colors
from pandas import plotting, DataFrame
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    dataset['data'], dataset['target'], random_state=0
)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

# create pandas dataframe from data in X_train
iris_dataframe = DataFrame(X_train, columns=dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = plotting.scatter_matrix(
    frame=iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=.8,
    cmap=colors.ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
)
pyplot.show()