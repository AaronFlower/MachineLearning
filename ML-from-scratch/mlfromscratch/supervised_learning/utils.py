import numpy as np


def divide_on_feature(X, feature_i, threshold):
    """Split data on feature i by threshold
    :X: Dataset
    :feature_i: feature i
    :threshold: value
    :returns: X1, X2
    """
    idx1 = X[:, feature_i] <= threshold
    idx2 = X[:, feature_i] > threshold
    return X[idx1], X[idx2]


def calc_variance(y):
    """Variance computed.
    :y: list or np.narray
    """
    return y.var()


def train_test_split(X, y, test_size=0.2):
    m, _ = X.shape
    idx = np.random.permutation(m)
    split_value = int(m * (1 - test_size))
    train_idx = idx[0:split_value]
    test_idx = idx[split_value:-1]

    train_X, train_y = X[train_idx], y[train_idx]
    test_X, test_y = X[test_idx], y[test_idx]
    return train_X, train_y, test_X, test_y


def mean_squared_error(y, y_hat):
    diff = (y - y_hat) ** 2
    return diff.mean()


def accuracy_score(y, y_hat):
    return np.sum(y == y_hat, axis=0) / len(y)


def to_categorical(y):
    m = np.shape(y)[0]
    col = np.max(y) + 1
    one_hot = np.zeros((m, col))
    one_hot[np.arange(m), y] = 1
    return one_hot
