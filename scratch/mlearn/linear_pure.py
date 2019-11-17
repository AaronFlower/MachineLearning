import numpy as np


class Linear(object):
    def __init__(self,
                 train_X,
                 train_y,
                 epochs=100,
                 learning_rate=0.01,
                 decay_lambda=0.1,
                 val_X=None,
                 val_y=None,
                 verbosity=1):
        self.X = train_X
        self.y = train_y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.decay_lambda = decay_lambda
        self.val_X = val_X
        self.val_y = val_y
        self.verbosity = verbosity

        _, n = train_X.shape
        self.w0 = np.zeros(1)
        self.weights = np.zeros((n, 1))

        if self.verbosity:
            print("Bias:", self.w0)
            print("Weights:", self.weights)

    def train(self):
        losses = []
        m, n = self.X.shape
        for i in range(self.epochs):
            pred = self.w0 + np.matmul(self.X, self.weights)
            diff = pred - self.y
            loss = 0.5 * np.mean(np.square(diff))
            self.w0 = self.w0 - self.learning_rate * np.mean(diff)
            self.weights = self.weights - (self.learning_rate / m) * np.matmul(self.X.T, diff)
            losses.append(loss)
            print("Epoch %d, loss %gf:" % ((i + 1), loss))

        return losses
