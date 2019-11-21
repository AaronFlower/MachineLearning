import numpy as np


def cross_entropy(y_score, y_true):
    epsilon = 1e-8
    return np.sum(-y_true * np.log(y_score + epsilon) - (1 - y_true) * np.log(1 - y_score + epsilon))


class FM(object):
    def __init__(self, epochs=1000, learning_rate=0.001, degree=2, verbosity=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.degree = degree
        self.verbosity = verbosity
        self.w0 = None
        self.W = None
        self.V = None

    def init_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        self.w0 = 0.0
        self.W = np.random.uniform(-limit, limit, n_features)
        self.V = np.random.uniform(-limit, limit, (n_features, self.degree))
        if self.verbosity:
            print("Init w0:", self.w0)
            print("Init W:", self.W)
            print("Init V:", self.V)

    def sigmoid(self, x):
        epsilon = 1e-5
        base = self.w0 + x.dot(self.W)

        s = 0
        for f in range(self.degree):
            left = 0
            right = 0
            for i, xi in enumerate(x):
                v_if = self.V[i, f]
                left += xi * v_if
                right += xi * xi * v_if * v_if
            s += (left * left - right)
        z = base + 0.5 * s
        # print('base = %g, s = %g,  z = %f' % (base, s, z))
        return 1.0 / (1 + np.exp(-z + epsilon))

    def predict(self, X):
        pass

    def train(self, X_train, y_train, X_val, y_val):
        m, n_features = X_train.shape
        self.init_weights(n_features)

        losses = []
        val_losses = []

        for e in range(self.epochs):
            loss = 0.0
            for x, y in zip(X_train, y_train):
                y_score = self.sigmoid(x)
                diff = y_score - y
                self.w0 -= self.learning_rate * diff
                self.W -= self.learning_rate * diff * x

                s = 0
                for f in range(self.degree):
                    for (i, xi) in enumerate(x):
                        s += xi * self.V[i, f]

                for (i, xi) in enumerate(x):
                    for f in range(self.degree):
                        self.V[i, f] -= self.learning_rate * diff * (xi * (s - xi * self.V[i, f]))

                loss += cross_entropy(y_score, y)

            loss = loss / m
            losses.append(loss)

            # if e % 10 == 0:
            #     print("V:", self.V)

            # y_score = self.sigmoid(X_train)
            # diff = y_score - y_train
            # self.w0 -= self.learning_rate * np.mean(diff)
            # self.W -= self.learning_rate * X_train.T.dot(diff)
            # self.V -= self.learning_rate * (X_train.T.dot(X_train.dot(self.V)) - np.sum(np.square(X_train).dot(self.V)))
            # # squared 可以缓存
            # loss = cross_entropy(y_score, y_train)
            # losses.append(loss)
            if self.verbosity:
                print("Epoch %d, loss: %g " % (e + 1, loss))

        return losses, val_losses, val_losses
