import numpy as np
from mlearn.pure.logistic import Base as LogisticRegression
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_breast_cancer(return_X_y=True)
y = y.reshape(-1, 1)

# Normalization
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

epochs = 1000
models = [
    {
        "name": "Alpha=0.03",
        "model": LogisticRegression(verbosity=0, epochs=epochs, learning_rate=0.03),
        "loss": None,
        "val": None,
        "acc": None,
    },

    {
        "name": "Alpha=0.1",
        "model": LogisticRegression(verbosity=0, epochs=epochs, learning_rate=0.1),
        "loss": None,
        "val": None,
        "acc": None,
    },

    {
        "name": "Alpha=0.5",
        "model": LogisticRegression(verbosity=0, epochs=epochs, learning_rate=0.5),
        "loss": None,
        "val": None,
        "acc": None,
    }
]


for m in models:
    m["loss"], m["val"] = m["model"].train(X_train, y_train, X_val, y_val)
    y_pred = m["model"].predict(X_test)

    count = 0
    for y_hat, y in zip(y_pred, y_test):
        if y_hat[0] == y[0]:
            count += 1
    m["acc"] = "Acc: %g" % (count * 1.0 / len(y_test) * 100)

x_epochs = np.arange(1, epochs + 1)

color_map = plt.get_cmap("tab10")
for i, m in enumerate(models):
    plt.plot(x_epochs, m['loss'], label=m["name"] + " Loss, " + m["acc"], color=color_map(i), linestyle="-")
    plt.plot(x_epochs, m['val'], label=m["name"] + " Val", color=color_map(i), linestyle="dotted")

plt.legend()
plt.show()

