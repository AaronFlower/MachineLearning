from mlearn.pure import poly2order
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_breast_cancer(return_X_y=True)
y = y.reshape(-1, 1)

# always note normalization
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min)/(X_max - X_min)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

epochs = 5000
m = poly2order.PolyRegression(epochs=epochs, learning_rate=0.001)
losses, val_losses = m.train(X_train, y_train, X_val, y_val)
y_pred = m.predict(X)
print("Acc = %g%%" % (np.sum(y_pred == y) * 1.0 / len(y) * 100))
x_epochs = np.arange(epochs)
plt.plot(x_epochs, losses, label="Loss")
plt.plot(x_epochs, val_losses, label="Val")
plt.legend()
plt.show()

