import numpy as np


def cross_entropy(y_pred, y):
    epsilon = 1e-8
    return np.mean(-y * np.log(y_pred + epsilon) - (1 - y) * np.log(1 - y_pred + epsilon))


def get_poly(X):
    m, n = X.shape
    print(m, n, int(n * (n - 1) / 2))
    X_poly = np.zeros((m, int(n * (n - 1) / 2)))
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            X_poly[:, count] = X[:, i] * X[:, j]
            count += 1
    return X_poly


class PolyRegression(object):
    def __init__(self, epochs=1000, learning_rate=0.001, verbosity=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.bias = None
        self.weights = None
        self.poly_weights = None

    def init_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        n_poly_features = int(n_features * (n_features - 1) / 2)
        self.bias = np.zeros(1)
        self.weights = np.random.uniform(-limit, limit, (n_features, 1))
        self.poly_weights = np.random.uniform(-limit, limit, (n_poly_features, 1))
        if self.verbosity:
            print("Bias:", self.bias)
            print("weights:", self.weights)
            print("poly_weights:", self.poly_weights)

    def sigmoid(self, X, X_poly):
        p = X_poly.dot(self.poly_weights)
        z = self.bias + X.dot(self.weights) + p
        return 1.0 / (1 + np.exp(-z))

    def predict(self, X):
        X_poly = get_poly(X)
        y_pred = self.sigmoid(X, X_poly)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred

    def train(self, X_train, y_train, X_val=None, y_val=None):
        m, n = X_train.shape
        X_train_poly = get_poly(X_train)
        if X_val is not None:
            X_val_poly = get_poly(X_val)
        if self.bias is None:
            self.init_weights(n)

        losses = []
        val_losses = []
        for i in range(self.epochs):
            y_train_pred = self.sigmoid(X_train, X_train_poly)
            diff = y_train_pred - y_train
            # loss = 0.5 * np.mean(np.square(diff))
            loss = cross_entropy(y_train_pred, y_train)
            self.bias = self.bias - self.learning_rate * np.mean(diff)
            self.weights = self.weights - self.learning_rate * X_train.T.dot(diff)
            self.poly_weights = self.poly_weights - self.learning_rate * X_train_poly.T.dot(diff)
            losses.append(loss)
            if self.verbosity:
                print("Epoch %d, loss: %g " % (i + 1, loss))

            if X_val is not None:
                y_val_pred = self.sigmoid(X_val, X_val_poly)
                # val_loss = 0.5 * np.mean(np.square(y_val_pred - y_val))
                val_loss = cross_entropy(y_val_pred, y_val)
                val_losses.append(val_loss)

        return losses, val_losses
