import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlearn.pure import fm
import matplotlib.pyplot as plt

X, y = datasets.load_breast_cancer(return_X_y=True)

# Data Normalization
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

y = y.reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Init models
epochs = 500
verbosity = 1

models = [
    {
        'name': 'LR, alpha = 0.01',
        'model': fm.FM(learning_rate=0.001, epochs=epochs, verbosity=verbosity),
        'loss': None,
        'val': None,
        'auc': None,
        'acc': None
    }
]

for m in models:
    m['loss'], m['val'],m['auc'] = m['model'].train(X_train, y_train, X_val, y_val)
    exit()
    y_pred = m['model'].predict(X_test)
    m['acc'] = (100.0 * np.sum(y_pred == y_test)) / len(y_test)
    print(m['acc'])

x_epochs = np.arange(epochs)


fig, axes = plt.subplots(2, 1)
cmap = plt.cm.get_cmap("Dark2")

for i, m in enumerate(models):
    axes[0].plot(x_epochs, m['loss'], label=m['name'] + " , loss", color=cmap(i))
    axes[0].plot(x_epochs, m['val'], label=m['name'] + " , val", linestyle="--", color=cmap(i))
    axes[1].plot(x_epochs, m['auc'], label=m['name'] + ' AUC', color=cmap(i))

axes[0].legend()
axes[1].legend()
plt.show()




