# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

dataset = datasets.load_diabetes()

# Using only one feature

X = dataset.data[:, np.newaxis, 2]
y = dataset.target

X_train = X[:-20]
X_test = X[-20:]

y_train = y[:-20]
y_test = y[-20:]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Coefficients: \n', model.coef_)
print('Mean squared error %.2f' %  mean_squared_error(y_pred, y_test))
print('Variance scroed %.2f' % r2_score(y_pred, y_test))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)

plt.show()
