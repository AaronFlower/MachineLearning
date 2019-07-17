import numpy as np
import matplotlib.pyplot as plt
from decision_tree import RegressionTree


def main():
    print('-- Regression Tree --')
    x = np.arange(1, 5)
    y = x + 1
    X = x.reshape(-1, 1)

    model = RegressionTree(0)
    model.fit(X, y)
    model.print_tree()

    test_X = np.array([1, 1.5, 1.999, 2.1, 4]).reshape(-1, 1)
    y_pred = model.predict(test_X)
    print(y_pred)
    cmap = plt.get_cmap('viridis')
    m1 = plt.scatter(X, y, s=10, color=cmap(0.9))
    m2 = plt.scatter(test_X, y_pred, color='k', s=10)
    plt.plot(test_X, y_pred, color='b', label='Decision Line')
    plt.legend((m1, m2),
               ("Training data", "Test data"), loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()
