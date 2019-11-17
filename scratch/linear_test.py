import numpy as np
import tensorflow as tf
from mlearn import linear
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_regression(n_features=500, bias=0.5)
# X, y = datasets.load_boston(return_X_y=True)
y.shape += (1, )

X = (X - X.min(axis=0))/X.max(axis=0)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).repeat().batch(200)
val = (val_X, val_y)

print(train_X.shape, train_y.shape)
print(val_X.shape, val_y.shape)
print(test_X.shape, test_y.shape)

epochs = 2000

models = {
    # "LinearNaive": {
    #     "model": linear.Linear(train_dataset, val=val, epochs=epochs, learning_rate=0.1, reg=None),
    #     "loss": None,
    #     "val": None,
    # },
    "Linear-L2": {
        "model": linear.Linear(train_dataset, val=val, epochs=epochs, learning_rate=1, reg="L2"),
        "loss": None,
        "val": None,
    }
}

for _, m in models.items():
    m['loss'], m['val'] = m['model'].train()
    print(m['loss'])
    print(m['val'])


x_epochs = np.arange(1, epochs + 1)

cmap = plt.get_cmap("tab10")
for i, (_, m) in enumerate(models.items()):
    plt.plot(x_epochs, m['loss'], label=m['model'].name() + " Loss", color=cmap(i), linestyle="-")
    plt.plot(x_epochs, m['val'], label=m['model'].name() + " Val", color=cmap(i), linestyle=":")

plt.legend()
plt.show()

