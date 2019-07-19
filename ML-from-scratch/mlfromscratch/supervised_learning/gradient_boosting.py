import numpy as np
from tqdm import tqdm
from decision_tree import RegressionTree
from loss_functions import CrossEntropy, SquareLoss, MAELoss


def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


class GradientBoosting(object):
    """
    Super class of GradientBoostingClassifier and GradientBoostingRegressor.
    Uses a collection of regression trees that trains on predicting the
    gradient of the loss function.

    Parameters:
    ----------
    n_estimators: int
        The number of classification trees that are uesd.
    learning_rate: float
        The step length that will be taken when following the negative gradeint
        during training
    min_samples_split: int
        The min number of samples needed to make a split when building a tree.
    min_impurity: float
        The min impurity required to split the tree further.
    max_depth: int
        The max depth of a tree
    regression: boolean
        True or false depending on if we're doing regression or classification
    """

    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression

        # square loss for regression
        # log loss for classification
        # self.loss = SquareLoss()
        self.loss = MAELoss()
        if not self.regression:
            self.loss = CrossEntropy()

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = RegressionTree(
                min_samples_split=min_samples_split,
                min_impurity=min_impurity,
                max_depth=max_depth
            )
            self.trees.append(tree)

    def fit(self, X, y):
        self.f0 = y.mean()
        y_pred = np.full(y.shape, y.mean())

        for i in tqdm(range(self.n_estimators)):
            residuals = y - y_pred
            gradient = np.sign(residuals)
            self.trees[i].fit(X, gradient, residuals=residuals)
            y_pred_i = np.array(self.trees[i].predict(X)).reshape(-1, 1)
            y_pred = y_pred + self.learning_rate * y_pred_i

    def predict(self, X):
        m, _ = X.shape
        y_pred = np.ones((m, 1)) * self.f0

        # Make predictions
        for tree in self.trees:
            y_pred_i = np.array(tree.predict(X)).reshape(m, 1)
            y_pred += self.learning_rate * y_pred_i

        if not self.regression:
            # Turn into probability distribution
            y_pred = np.exp(y_pred) / np.sum(
                np.exp(y_pred), axis=1).reshape(-1, 1)
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GradientBoostingRegressor(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=0.5,
                 min_samples_split=2, min_impurity=1e-7,
                 max_depth=4):
        GradientBoosting.__init__(self,
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  min_samples_split=min_samples_split,
                                  min_impurity=min_impurity,
                                  max_depth=max_depth,
                                  regression=True
                                  )


class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, n_estimators=200, learning_rate=0.5,
                 min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2):
        GradientBoosting.__init__(self,
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  min_samples_split=min_samples_split,
                                  min_impurity=min_info_gain,
                                  max_depth=max_depth,
                                  regression=True)

    def fit(self, X, y):
        y = to_categorical(y)
        GradientBoosting.fit(self, X, y)
