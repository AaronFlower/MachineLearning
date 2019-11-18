import numpy as np


def cross_entropy(y_hat, y):
    epsilon = 1e-8
    return np.mean(-y * np.log(y_hat + epsilon) - (1 - y) * np.log(1 - y_hat + epsilon))


class Base(object):
    def __init__(self, epochs=1000, learning_rate=0.01, verbosity=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.bias = None
        self.weights = None

    def init_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        self.bias = np.zeros(1)
        self.weights = np.random.uniform(-limit, limit, (n_features, 1))
        if self.verbosity:
            print("Init Bias:", self.bias)
            print("Init Weights:", self.weights)

    def sigmoid(self, X):
        z = self.bias + np.matmul(X, self.weights)
        return 1.0 / (1 + np.exp(-z))

    def predict(self, X):
        y_hat = self.sigmoid(X)
        y_hat[y_hat >= 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        return y_hat

    def train(self, X_train, y_train, X_val, y_val):
        m, n = X_train.shape
        self.init_weights(n)
        assert y_train.shape == (m, 1)
        losses = []
        val_losses = []
        for i in range(self.epochs):
            y_hat = self.sigmoid(X_train)
            loss = cross_entropy(y_hat, y_train)
            diff = y_hat - y_train
            losses.append(loss)
            self.bias = self.bias - self.learning_rate * np.mean(diff)
            self.weights = self.weights - self.learning_rate * X_train.T.dot(diff)

            if X_val is not None:
                val_hat = self.sigmoid(X_val)
                val_loss = cross_entropy(val_hat, y_val)
                val_losses.append(val_loss)

        return losses, val_losses
