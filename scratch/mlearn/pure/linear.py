import numpy as np


class NoRegularization(object):
    def __init__(self):
        pass

    def __call__(self, w):
        return 0

    def grad(self, w):
        return 0


class L2Regularization(object):
    def __init__(self, decay_ratio=0.5):
        self.decay_ratio = decay_ratio

    def __call__(self, w):
        return np.sum(0.5 * self.decay_ratio * np.square(w))

    def grad(self, w):
        return self.decay_ratio * w


class Base(object):
    def __init__(self,
                 epochs=100,
                 learning_rate=0.01,
                 verbosity=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.w0 = None
        self.weights = None
        # self.regularization = lambda x: 0
        # self.regularization.grad = lambda x: 0
        self.regularization = NoRegularization()
        self.norm_axis = None

    def init_weights(self, n_features):
        limit = 1 / np.sqrt(n_features)
        self.w0 = np.zeros(1)
        self.weights = np.random.uniform(-limit, limit, (n_features, 1))

        if self.verbosity:
            print("Bias:", self.w0)
            print("Weights:", self.weights)

    def normalize(self, X):
        if X is None:
            return

        if self.norm_axis is None:
            self.norm_axis = {
                "min": X.min(axis=0),
                "max": X.max(axis=0)
            }
        return (X - self.norm_axis["min"]) / (self.norm_axis["max"] - self.norm_axis["min"])

    def loss(self, diff):
        return 0.5 * np.mean(np.square(diff)) + self.regularization(self.weights)

    def train(self, train_X, train_y, val_X=None, val_y=None):
        m, n = train_X.shape
        assert train_y.shape == (m, 1)
        self.init_weights(n_features=n)
        losses = []
        val_losses = []
        for i in range(self.epochs):
            y_pred = self.w0 + np.matmul(train_X, self.weights)
            diff = y_pred - train_y
            loss = self.loss(diff)
            self.w0 = self.w0 - self.learning_rate * np.mean(diff)
            grad = (1 / m) * np.matmul(train_X.T, diff) + self.regularization.grad(self.weights)
            self.weights = self.weights - self.learning_rate * grad
            losses.append(loss)

            if val_X is not None:
                val_pred = self.w0 + np.matmul(val_X, self.weights)
                val_loss = self.loss(val_pred - val_y)
                val_losses.append(val_loss)

            if self.verbosity:
                print("Epoch %d, loss %gf:" % ((i + 1), loss))

        return losses, val_losses

    def predict(self, X):
        X = self.normalize(X)
        return self.w0 + np.matmul(X, self.weights)


class Linear(Base):
    def __init__(self, epochs=1000, learning_rate=0.001, verbosity=1):
        super(Linear, self).__init__(epochs, learning_rate, verbosity)

    def train(self, train_X, train_y, val_X=None, val_y=None):
        train_X = self.normalize(train_X)
        val_X = self.normalize(val_X)
        return super(Linear, self).train(train_X, train_y, val_X, val_y)


class LinearL2(Base):
    def __init__(self, epochs=1000, learning_rate=0.001, verbosity=1, decay_ratio=0.1):
        super(LinearL2, self).__init__(epochs, learning_rate, verbosity)
        self.regularization = L2Regularization(decay_ratio)

    def train(self, train_X, train_y, val_X=None, val_y=None):
        train_X = self.normalize(train_X)
        val_X = self.normalize(val_X)
        return super(LinearL2, self).train(train_X, train_y, val_X, val_y)
