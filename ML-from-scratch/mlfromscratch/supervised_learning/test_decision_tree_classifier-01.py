# -*- coding: utf-8 -*-

"""
机器学习实战上的海洋鱼类的数据集
"""

import numpy as np
from decision_tree import ClassificationTree


# 创建数据集
def loadDataSet():
    data = [
        [1, 1],
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 1]
    ]
    y = [1, 1, 0, 0, 0]
    labels = ['yes', 'yes', 'no', 'no', 'no']
    features = ['no surfacing', 'flippers']
    return np.array(data), np.array(y).reshape(-1, 1), labels, features


def main():
    print('-- Classification Tree --')
    X, y, labels, features = loadDataSet()

    model = ClassificationTree()
    model.fit(X, y)
    model.print_tree()
    y_pred = model.predict(X)
    y_pred = np.array(y_pred).reshape(-1, 1)

    err_rate = np.sum(y != y_pred) / len(y)
    print('Error rate is {0:0.2f}%'.format(err_rate))


if __name__ == "__main__":
    main()
