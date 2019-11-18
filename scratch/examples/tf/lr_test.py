import tensorflow as tf
import numpy as np
from mlearn.tf.lr import LRModel
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_classification(n_samples=1000, n_features=100, n_clusters_per_class=1)
X, y = datasets.load_breast_cancer(return_X_y=True)

y = y.reshape(-1, 1)

# Normalization
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

y = y.astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(2000).batch(400)

epochs = 1000
lr = LRModel(verbosity=0, epochs=epochs)
print(lr.name)

losses, val_losses = lr.train(train_dataset, X_val, y_val)

y_pred = lr.predict(X_test)
print(X_test)
print(y_pred)
print(list(zip(y_test, y_pred)))

count = 0
for y_hat, y in zip(y_pred, y_test):
    if y_hat[0] == y[0]:
        count += 1

print("Acc : %g" % (count * 1.0 / len(y_test) * 100))

x_epochs = np.arange(epochs)
plt.plot(x_epochs, losses, label="loss")
plt.plot(x_epochs, val_losses, label="val")
plt.legend()
plt.show()
