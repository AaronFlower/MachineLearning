# -*- coding: utf-8 -*-

import numpy as np

# Reference: https://en.wikipedia.org/wiki/Activation_function


class Sigmoid():

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class ReLU():

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self,  x):
        return np.where(x >= 0, 1, self.alpha)
