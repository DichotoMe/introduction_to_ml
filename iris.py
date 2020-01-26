from sklearn.datasets import load_iris
dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    dataset['data'], dataset['target'], random_state=0
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

