from mlearn.pure import poly2order
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 2],
    [1, 0],
    [-1, 1],
    [-0.5, 1],
    [0.5, 1],
    [0.5, 0.8],
    [0.5, 0.7]
], dtype=np.float)

y = np.array([0, 1, 1, 1, 0, 0, 0])
y = y.reshape(-1, 1)

# X = np.arange(6).reshape(2, 3)
print(X)
X_poly = poly2order.get_poly(X)
print(X_poly)

epochs = 1000
m = poly2order.PolyRegression(epochs=epochs, learning_rate=0.1)
losses, val_losses = m.train(X, y, X, y)
y_pred = m.predict(X)
print("Acc = %g%%" % (np.sum(y_pred == y) * 1.0 / len(y) * 100))
x_epochs = np.arange(epochs)
plt.plot(x_epochs, losses, label="Loss")
plt.plot(x_epochs, val_losses, label="Val")
plt.legend()
plt.show()

