# -*- coding: utf-8 -*-

from sklearn import datasets
from xgboost import XGBoost
from utils import train_test_split, accuracy_score


def main():
    print('-- XGBoost --')

    X, y = datasets.load_iris(return_X_y=True)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3)

    clf = XGBoost(n_estimators=2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(y_pred)
    print(y_test)
    accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy :", accuracy)


if __name__ == "__main__":
    main()
