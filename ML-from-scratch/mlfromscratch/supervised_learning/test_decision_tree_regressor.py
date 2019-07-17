import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decision_tree import RegressionTree


def train_test_split(X, y, test_size=0.2):
    m, _ = X.shape
    idx = np.random.permutation(m)
    split_value = int(m * (1 - test_size))
    train_idx = idx[0:split_value]
    test_idx = idx[split_value:-1]

    train_X, train_y = X[train_idx], y[train_idx]
    test_X, test_y = X[test_idx], y[test_idx]
    return train_X, train_y, test_X, test_y


def mean_squared_error(y, y_hat):
    diff = (y - y_hat) ** 2
    return diff.mean()


def main():
    print("--- Regression Tree --")

    data = pd.read_csv('./TempLinkoping2016.txt', sep='\t')
    time = data['time'].values.reshape(-1, 1)
    temp = data['temp'].values.reshape(-1, 1)

    X = (time - time.mean()) / time.std()
    y = temp[:, 0]      # Temperature. Reduce to one-dim

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.3)

    model = RegressionTree(3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # y_pred_line = model.predict(X)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    model.print_tree()

    # Color map
    cmap = plt.get_cmap('viridis')
    m1 = plt.scatter(366 * X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * X_test, y_test, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * X_test, y_pred, color='black', s=10)
    # m4 = plt.scatter(366 * X, y_pred_line, color='red', s=10)
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3),
               ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
