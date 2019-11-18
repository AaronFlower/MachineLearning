import tensorflow as tf
import numpy as np
from mlearn.tf.lr import LRModel

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=100, n_clusters_per_class=1)

X = (X - X.min(axis=0))/X.max(axis=0)

y.shape += (1,)

y = y.astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().shuffle(20000).batch(400)

print(X_train.shape)
print(y_train.shape)

lr = LRModel(train_dataset, verbosity=1, epochs=1000)
print(lr.name)

lr.train()

pred = lr.predict(X_test)
y_hat = np.zeros((len(y_test), 1))

pos_indices = pred >= 0.5
y_hat[pos_indices] = 1

count = 0
for y_pred, y in zip(y_hat, y_test):
    if y_pred[0] == y[0]:
        count += 1

# 过拟合比较严重
print("Acc : %.3f" % (count * 1.0 / len(y_test)))
