# -*- coding: utf-8 -*-

"""
机器学习实战上的隐形眼睛数据
"""

import pandas as pd
import numpy as np
from decision_tree import ClassificationTree

def loadDataSet():
    df = pd.read_csv('./lenses.txt', sep='\t', header=None)
    for i in df:
        df[i] = pd.factorize(df[i])[0]
    data = df.to_numpy()
    X = data[:, 0:-1]
    y = data[:, -1:]
    return X, y

def main():
    print('-- Classification Tree --')
    X, y = loadDataSet()

    model = ClassificationTree()
    model.fit(X, y)
    model.print_tree()
    y_pred = model.predict(X)
    y_pred = np.array(y_pred).reshape(-1, 1)

    err_rate = np.sum(y != y_pred) / len(y)
    print('Error rate is {0:0.2f}%'.format(err_rate))


if __name__ == "__main__":
    main()
