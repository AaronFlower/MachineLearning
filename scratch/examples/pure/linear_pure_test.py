from mlearn.pure import linear
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# y = x + 0.5
train_X = np.array([[1], [2], [3], [4], [5]])
train_y = np.array([[1.5], [2.5], [3.5], [4.5], [5.5]])

model = linear.Linear(epochs=1000, learning_rate=0.1)

model.train(train_X, train_y)

# z = 2x + y + 1
train_X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0.5, 0.5],
    [-1, 0],
    [-1, 1],
])
train_y = np.array([
    [1],
    [2],
    [3],
    [4],
    [2.5],
    [-1],
    [0],
])

model = linear.Linear(epochs=1000, learning_rate=0.1)

model.train(train_X, train_y)

y_pred = model.predict(train_X)

X, y = datasets.load_boston(return_X_y=True)
X = (X - X.min(axis=0))/X.max(axis=0)
print(X[0:2])
print(y[0:2])
y = y.reshape(-1, 1)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

model = linear.Linear(epochs=1000, learning_rate=0.5)
model.train(train_X, train_y)

modelL2 = linear.LinearL2(epochs=1000, learning_rate=0.01, verbosity=1, decay_ratio=0.2)
modelL2.train(train_X, train_y)


