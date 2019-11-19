from mlearn.tf import multiclass as model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = np.array([
    [1, 2],
    [1, 0],
    [-1, 1],
    [-0.5, 1],
    [0.5, 1],
    [0.5, 0.8],
    [0.5, 0.7]
], dtype=np.float)

n_classes = 2
y = np.array([0, 1, 1, 1, 0, 0, 0])

y_hot = np.zeros((len(y), n_classes))

for i, label in enumerate(y):
    y_hot[i, label] = 1

dataset = tf.data.Dataset.from_tensor_slices((X, y_hot)).shuffle(1000).repeat().batch(10)

epochs = 1000
m = model.MulticlassLinear(epochs=epochs, verbosity=0)
losses, val_losses = m.train(dataset, X, y_hot)
print(losses)
print(val_losses)
y_pred = m.predict(X)
print("Acc : %g" % (1.0 * np.sum(y == y_pred) / len(y) * 100))

x_epochs = np.arange(epochs)
fig, axes = plt.subplots(2, 1)
print(losses)
print(val_losses)
axes[0].plot(x_epochs, losses, label="Loss")
axes[0].plot(x_epochs, val_losses, label="val")

plt.legend()
plt.show()

