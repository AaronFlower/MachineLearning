# import numpy as np
# import matplotlib.pyplot as plt
from sklearn import datasets

from gradient_boosting import GradientBoostingClassifier as GBClf
from utils import train_test_split, accuracy_score


def main():
    print("-- Gradient Boosting Classification --")
    data = datasets.load_iris()
    X = data.data
    y = data.target

    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    clf = GBClf(n_estimators=200)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {0:0.3f}%".format(accuracy * 100))

    # Todo plot


if __name__ == "__main__":
    main()
