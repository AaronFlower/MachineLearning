import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score


def cross_entropy(y_score, y):
    epsilon = 1e-8
    return np.mean(-y * np.log(y_score + epsilon) - (1 - y) * np.log(1 - y_score + epsilon))


def init_lr_weights(n_features):
    limit = 1 / np.sqrt(n_features)
    weights = {
        "bias": np.zeros(1),
        "W": np.random.uniform(-limit, limit, (n_features, 1))
    }
    return weights


class BaseLR(object):
    def __init__(self, epochs=1000, learning_rate=0.001, verbosity=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.weights = None
        self.transformX = lambda X: X

    def init_weights(self, n_features):
        self.weights = init_lr_weights(n_features)
        if self.verbosity:
            print("Init Weights:", self.weights)

    def sigmoid(self, X):
        z = self.weights['bias'] + X.dot(self.weights['W'])
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        X = self.transformX(X)
        y_score = self.sigmoid(X)
        y_pred = np.zeros(y_score.shape)
        y_pred[y_score >= 0.5] = 1
        return y_pred

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # 数据 transform
        X_train = self.transformX(X_train)
        X_val = self.transformX(X_val)

        _, n_features = X_train.shape
        self.init_weights(n_features)

        losses = []
        val_losses = []
        aucs = []

        for i in range(self.epochs):
            y_score = self.sigmoid(X_train)
            diff = y_score - y_train
            self.weights['bias'] -= self.learning_rate * np.mean(diff)
            self.weights['W'] -= self.learning_rate * X_train.T.dot(diff)

            loss = cross_entropy(y_score, y_train)
            losses.append(loss)
            if self.verbosity:
                print("Epoch %d, loss: %g" % (i + 1, loss))

            if X_val is not None:
                y_score = self.sigmoid(X_val)
                loss = cross_entropy(y_score, y_val)
                val_losses.append(loss)
                auc = roc_auc_score(y_val, y_score)
                aucs.append(auc)

        return losses, val_losses, aucs


class PolyLR(BaseLR):
    def __init__(self, epochs=1000, learning_rate=0.001, verbosity=1, degree=2, interaction_only=True):
        super(PolyLR, self).__init__(epochs=epochs, learning_rate=learning_rate, verbosity=verbosity)
        poly = PolynomialFeatures(degree=degree,
                                  include_bias=False,
                                  interaction_only=interaction_only)
        self.transformX = poly.fit_transform



