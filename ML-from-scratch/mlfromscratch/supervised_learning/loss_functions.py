import numpy as np


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        return NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class SquareLoss(Loss):

    def __init__(self):
        Loss.__init__(self)

    def loss(self, y, y_pred):
        return 0.5 * ((y - y_pred) ** 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)


class MAELoss(Loss):

    def __init__(self):
        Loss.__init__(self)

    def gradient(self, y, y_pred):
        return np.sign(y - y_pred)

    def loss(self, y, y_pred):
        return NotImplementedError()


class CrossEntropy(Loss):

    def __init__(self):
        Loss.__init__(self)

    def loss(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)

    def gradient(self, y, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y/y_pred) + (1 - y)/(1 - y_pred)
