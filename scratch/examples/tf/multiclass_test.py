from mlearn.tf import multiclass as model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(len(train_images), -1)
test_images = test_images.reshape(len(test_images), -1)

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.33)

train_labels = tf.one_hot(train_labels, depth=10, dtype=tf.float64)
val_labels = tf.one_hot(val_labels, depth=10, dtype=tf.float64)
fashion_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).repeat().batch(1000)

epochs = 5000
m = model.MulticlassLinear(epochs=epochs, verbosity=0)
losses, val_losses = m.train(fashion_dataset, val_images, val_labels)
y_pred = m.predict(test_images)
print("Acc : %g" % (1.0 * np.sum(test_labels == y_pred) / len(y_pred) * 100))

x_epochs = np.arange(epochs)
fig, axes = plt.subplots(2, 1)
print(losses)
print(val_losses)
axes[0].plot(x_epochs, losses, label="Loss")
axes[0].plot(x_epochs, val_losses, label="val")
axes[0].legend()

plt.show()
