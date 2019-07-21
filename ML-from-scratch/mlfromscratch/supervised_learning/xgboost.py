# -*- coding: utf-8 -*-

"""
XGBoost 实现分类，multi-class 使用 softmax 函数。操作有点麻烦呀。
"""

import numpy as np
from decision_tree import XGBoostRegressionTree
from tqdm import tqdm
from activation_functions import Sigmoid
from utils import to_categorical


class LogisticLoss():

    def __init__(self):
        sigmoid = Sigmoid()
        self.log_func = sigmoid
        self.log_grad = sigmoid.gradient

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # p = self.log_func(y_pred)
        return -y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)
        # return y * np.log(p) + (1 - y) * np.log(1 - p)

    def gradient(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y/y_pred + (1 - y)/(1 - y_pred)
        # return -(y - p)

    def hess(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y / (y_pred ** 2) - (1 - y) / ((1 - y_pred) ** 2)


class XGBoost(object):

    """
    The XGBoost classifier

    Parameters:
    ----------
    n_estimators: int
        Then number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradeint
        during training.
    min_sample_split: int
        The minimun number of samples needed to make a split when building
        a tree.
    min_purity: float
        The minimum impurity required to split the tree further
    max_depth: int
        The maximum depth of a tree.
    """

    def __init__(self, n_estimators=200, learning_rate=0.001,
                 min_samples_split=2, min_impurity=1e-7, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth

        # Log loss for classification
        self.loss = LogisticLoss()

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_impurity,
                max_depth=self.max_depth,
                loss=self.loss
            )
            self.trees.append(tree)

    def fit(self, X, y):
        y = to_categorical(y)

        y_pred = np.zeros(np.shape(y))

        for i in tqdm(range(self.n_estimators)):
            tree = self.trees[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred)
            update_pred = tree.predict(X)

            y_pred -= np.multiply(self.learning_rate, update_pred)

    def predict(self, X):
        y_pred = None
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred -= np.multiply(self.learning_rate, update_pred)

        # Turn into probability distribution (Softmax)
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
