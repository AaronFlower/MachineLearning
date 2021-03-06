# -*- coding: utf-8 -*-

"""
Code Reference: https://github.com/eriklindernoren/ML-From-Scratch
树的实现并不要求传入的样本集 X, y 是 np.narray 的类型，所以在编码计算时使用很多
np 的方法，如: np.shape(), np.expand_dims(), 包括在计算 var, mean 时，这样实现
起来很不方便，如是不能向量化操作的话，性能也会变的很差。所在自己实现中，保证操
作的类型是 np.narray 类型很重要。

- [ ] Only support np.narray

"""

import numpy as np
import math


def log2(x):
    return math.log(x) / math.log(2)


def calc_entropy(y):
    """ Calculate the entropy of y """
    values = np.unique(y)
    n = len(y)
    entropy = 0
    for v in values:
        count = len(y[y == v])
        p = count / n
        entropy -= p * log2(p)
    return entropy


def divide_on_feature(X, feature_i, threshold):
    """Split data on feature i by threshold
    :X: Dataset
    :feature_i: feature i
    :threshold: value
    :returns: X1, X2
    """
    idx1 = X[:, feature_i] <= threshold
    idx2 = X[:, feature_i] > threshold
    return X[idx1], X[idx2]


class DecisionNode():
    '''Class that represents a decision node or leaf in the decision tree

    Parameters:
        - feature_i: int
            Feature index which we want to use as the threshold measure
        - threshold: float
            The value that we will compare feature values at feature_i
            against to determine the prediction
        - value: float
            The class label if classification tree,or float if regression tree.
        - true_branch: DecisionNode
            Next decision node for samples where features met the threshold
        - false_branch: DecisionNode
            Next decision node for samples where features did not met threshold
    '''
    def __init__(self, feature_i=None, threshold=None, value=None,
                 true_branch=None, false_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch


class DecisionTree(object):
    '''Super class of RegressionTree and ClassificationTree.

    Parameters:
        - min_samples_split: int
            the min number of samples needed to make a split.
        - min_impurity: float
            the min impurity required to split the tree further
        - max_depth: int
            the max depth of a tree
        - loss: function
            loss function is used for Gradient Boosting models for calculate
            impurity.
    '''
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float('inf'), loss=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_calc = None      # 划分准则
        self._leaf_value_calc = None    # 叶子生成准则
        # If y is one-hot encoded (multi-dim) or not(one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.loss = loss

    def fit(self, X, y, loss=None, redisuals=None):
        """ Build decision tree  """
        self.one_dim = len(np.shape(y)) == 1    # as y may not is np.narray
        self.root = self._build_tree(X, y, redisuals=redisuals)
        self.loss = None

    def _build_tree(self, X, y, cur_depth=0, redisuals=None):
        """Recursive method which builds out the decision tree and
        splits X and  respective y on the feature of X which (based on
        impurity) best separates the data
        """
        largest_impurity = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None

        # Check if expansion of is needed
        if len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and cur_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Iterate through all unique values of feature column i and
                # calcuate the impurity
                for threshold in unique_values:
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # y1 = Xy1[:, -1:]  # The difference between Xy1[:, -1]
                        # y2 = Xy2[:, -1:]
                        # you cant use y1 = Xy1[:, -1:] because the y is multi
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calcualte impurity
                        impurity = self._impurity_calc(y, y1, y2)

                        # If this threshold resulted in a higher information
                        # gain than previously recorded save the threshold
                        # value and the feature index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {
                                "feature_i": feature_i,
                                "threshold": threshold
                            }
                            best_sets = {
                                "leftX": Xy1[:, :n_features],
                                "lefty": y1,
                                "rightX": Xy2[:, :n_features],
                                "righty": y2
                            }

        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets["leftX"],
                                           best_sets["lefty"], cur_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"],
                                            best_sets["righty"], cur_depth + 1)
            return DecisionNode(feature_i=best_criteria['feature_i'],
                                threshold=best_criteria['threshold'],
                                true_branch=true_branch,
                                false_branch=false_branch)

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calc(y)
        if redisuals is not None:
            leaf_value = np.median(y)

        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        """Do a recursive search down the tree and make a prediction vlaue of
        leaf that we end up at
        """
        if tree is None:
            tree = self.root

        if tree.value is not None:
            return tree.value

        feature_value = x[tree.feature_i]
        branch = tree.false_branch
        if feature_value <= tree.threshold:
            branch = tree.true_branch

        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """
        Recursively print the decision tree
        """
        if tree is None:
            tree = self.root

        if tree.value is not None:
            print("{0:.3f}".format(tree.value))
        else:
            print("{0}:{1:.1f}".format(tree.feature_i, tree.threshold))
            print(indent + "T->", end="")
            self.print_tree(tree.true_branch, indent + indent)
            print(indent + "F->", end="")
            self.print_tree(tree.false_branch, indent + indent)


class RegressionTree(DecisionTree):

    """RegressionTree for Gradient Boosting"""

    def _calc_variance(self, y):
        '''
        https://stattrek.com/matrix-algebra/covariance-matrix.aspx#Problem1
        一维向量可以很容易计算相应的方差，对于多维向量。计算时应使用矩阵的
        variance-covariance 来计算，最后对对角线上的元素求和即可。
        '''
        deviation = y - y.mean(axis=0)
        covariance = deviation.T.dot(deviation) / len(y)
        return np.diag(covariance)

    def _calc_variance_reduction(self, y, y1, y2):
        # https://stattrek.com/matrix-algebra/covariance-matrix.aspx#Problem1
        # 一维向量可以很容易计算相应的方差，对于多维向量。计算时应使用矩阵的
        # variance-covariance 来计算，最后对对角线上的元素求和即可。
        y_var = self._calc_variance(y)
        y1_var = self._calc_variance(y1)
        y2_var = self._calc_variance(y2)

        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        return sum(y_var - (frac_1 * y1_var + frac_2 * y2_var))

    def _mean_of_y(self, y):
        # 应该注意目标可能是 one-hot 的
        value = y.mean(axis=0)
        return value[0] if self.one_dim else value

    def fit(self, X, y, residuals=None):
        self._impurity_calc = self._calc_variance_reduction
        self._leaf_value_calc = self._mean_of_y
        super(RegressionTree, self).fit(X, y, residuals)


class ClassificationTree(DecisionTree):

    """Classification Tree"""

    # 不写构造函数，就会默认调用父类构造函数。
    # def __init__(self, max_depth=float('inf')):
    #     """ """
    #     DecisionTree.__init__(self, max_depth=max_depth)

    def _calc_info_gain(self, y, y1, y2):
        ent_y = calc_entropy(y)
        ent_y1 = calc_entropy(y1)
        ent_y2 = calc_entropy(y2)

        frac_1 = len(y1)/len(y2)
        frac_2 = 1 - frac_1
        info_gain = ent_y - (frac_1 * ent_y1 + frac_2 * ent_y2)
        return info_gain

    def _majority_vote(self, y):
        label = None
        max_count = 0
        values = np.unique(y)
        for v in values:
            count = len(y[y == v])
            if count >= max_count:
                count = max_count
                label = v
        return label

    def fit(self, X, y):
        self._impurity_calc = self._calc_info_gain
        self._leaf_value_calc = self._majority_vote
        DecisionTree.fit(self, X, y)


class XGBoostRegressionTree(DecisionTree):

    """XGBoostRegressionTree"""

    def _split(self, y):
        """
        y contains y_true in left half of the middle column and
        y_pred in the right half.
        Split and return the two matrices
        """
        col = int(np.shape(y)[1]/2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def _gain(self, y, y_pred):
        gi = self.loss.gradient(y, y_pred)
        hi = self.loss.hess(y, y_pred)
        print(gi)
        print(hi)
        nominator = np.power((y * gi).sum(), 2)
        denominator = hi.sum()
        return 0.5 * (nominator / denominator)

    def _gain_by_taylor(self, y, y1, y2):
        # Split to compute impurity
        y, y_pred = self._split(y)
        y1, y1_pred = self._split(y1)
        y2, y2_pred = self._split(y2)

        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return true_gain + false_gain - gain

    def _approximate_update(self, y):
        # to evaluate the leaf
        # y split into y, y_pred
        y, y_pred = self._split(y)
        # Newton's Method
        gi = np.sum(y * self.loss.gradient(y, y_pred), axis=0)
        hi = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation = hi / gi
        return update_approximation

    def fit(self, X, y):
        self._impurity_calc = self._gain_by_taylor
        self._leaf_value_calc = self._approximate_update
        DecisionTree.fit(self, X, y)
