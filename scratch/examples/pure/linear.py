import numpy as np
from mlearn.pure import linear
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

learning_rate = 0.1
epochs = 100
verbosity = 0

models = [
    {
        "name": "Simple, alpha=0.1",
        "model": linear.Linear(epochs=epochs, learning_rate=learning_rate, verbosity=verbosity),
        "loss": None,
        "val": None,
    },
    {
        "name": "Simple, alpha=0.5",
        "model": linear.Linear(epochs=epochs, learning_rate=0.5, verbosity=verbosity),
        "loss": None,
        "val": None,
    },
    {
        "name": "L2, alpha=0.1",
        "model": linear.LinearL2(epochs=epochs, learning_rate=learning_rate, decay_ratio=0.05, verbosity=verbosity),
        "loss": None,
        "val": None,
    }
]

X, y = datasets.load_boston(return_X_y=True)
# X, y = datasets.load_diabetes(return_X_y=True)
# X, y = datasets.make_regression(n_samples=1000, n_features=100, n_informative=10)
y = y.reshape(-1, 1)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.2)

for m in models:
    m["loss"], m["val"], m['aucs'] = m["model"].train(train_X, train_y, val_X, val_y)


x_epochs = np.arange(1, epochs + 1)

color_map = plt.get_cmap("tab10")
for i, m in enumerate(models):
    plt.plot(x_epochs, m['loss'], label=m["name"] + " Loss", color=color_map(i), linestyle="-")
    plt.plot(x_epochs, m['val'], label=m["name"] + " Val", color=color_map(i), linestyle=":")


plt.legend()
plt.show()
