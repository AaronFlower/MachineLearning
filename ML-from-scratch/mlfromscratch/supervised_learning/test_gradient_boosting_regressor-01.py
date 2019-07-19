import pandas as pd
import matplotlib.pyplot as plt

from gradient_boosting import GradientBoostingRegressor as GBReg
from utils import train_test_split, mean_squared_error


def main():
    data = pd.read_csv('./TempLinkoping2016.txt', sep='\t')
    data = data.to_numpy()
    X = data[:, 0:-1]
    y = data[:, -1:]
    X = (X - X.mean()) / X.std()

    train_X, train_y, test_X, test_y = train_test_split(X, y)

    model = GBReg(n_estimators=200, learning_rate=0.5)
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)

    mse = mean_squared_error(test_y, y_pred)
    print('Mean Squared Error:', mse)

    # Color map
    cmap = plt.get_cmap('viridis')
    m1 = plt.scatter(366 * train_X, train_y, color=cmap(0.9), s=10)
    m2 = plt.scatter(366 * test_X, test_y, color=cmap(0.5), s=10)
    m3 = plt.scatter(366 * test_X, y_pred, color='black', s=10)
    plt.title("MSE: %.2f" % mse, fontsize=10)
    plt.xlabel('Day')
    plt.ylabel('Temperature in Celcius')
    plt.legend((m1, m2, m3),
               ("Training data", "Test data", "Prediction"), loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
