from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def plot_training_vs_test_accuracy_by_neighbors(X, y, method_class, neighbors_range, stratify=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=stratify, random_state=66)

    training_accuracy = []
    test_accuracy = []

    for n_neighbors in neighbors_range:
        reg = method_class(n_neighbors).fit(X_train, y_train)

        training_accuracy.append(reg.score(X_train, y_train))
        test_accuracy.append(reg.score(X_test, y_test))

    plt.plot(neighbors_range, training_accuracy, label="training accuracy")
    plt.plot(neighbors_range, test_accuracy, '--', label="test accuracy")
    plt.xlabel("n_neighbors")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()
