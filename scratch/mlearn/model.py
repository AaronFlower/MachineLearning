"""
Define Base Model
"""


class Model(object):
    """
    Base Model
    """

    def __init__(self):
        self.name = "base model"

    def predict(self, X):
        pass

    def loss(self, y_pred, y):
        pass

    def train(self, X, y):
        pass
