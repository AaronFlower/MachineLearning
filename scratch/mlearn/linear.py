import tensorflow as tf
import numpy as np


def mse_loss(y_hat, y):
    return 0.5 * tf.reduce_mean(tf.square(y_hat - y))


class Linear(object):
    def __init__(self,
                 dataset,
                 val=None,
                 epochs=100,
                 learning_rate=0.01,
                 reg="L2",
                 decay_lambda=0.1,
                 optimizer=None,
                 verbosity=0
                 ):
        self.dataset = dataset
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg = reg
        if self.reg:
            self.reg = reg.upper()

        self.decay_lambda = decay_lambda
        self.verbosity = verbosity
        self.optimizer = optimizer
        self.val = val

        if self.optimizer is None:
            # SGD does not converge, why?
            # self.optimizer = tf.optimizers.SGD(self.learning_rate)
            self.optimizer = tf.optimizers.Adagrad(self.learning_rate)

        data = dataset.take(1)
        for X, _ in data:
            _, self.n_features = X.shape

        self.bias = tf.Variable(np.zeros(1), dtype=tf.float64, name="bias")
        self.weights = tf.Variable(np.zeros((self.n_features, 1)), dtype=tf.float64, name="weights")

        if self.verbosity:
            print("Linear Regression Model Initialization")
            print("Bias: ", self.bias)
            print("Weights: ", self.weights)

    def name(self):
        name = "Linear: alpha = %g, lambda = %g" % (self.learning_rate, self.decay_lambda)
        if self.reg:
            name += ", reg = " + self.reg
        return name

    def predict(self, X):
        y_hat = self.bias + tf.matmul(X, self.weights)
        return y_hat

    def l2_reg(self):
        return 0.5 * self.decay_lambda * tf.matmul(tf.transpose(self.weights), self.weights)

    def train(self):
        losses = []
        val_losses = []
        dataset = self.dataset.take(self.epochs)

        if self.val:
            val_X, val_y = self.val

        for i, (X, y) in enumerate(dataset):
            with tf.GradientTape() as g:
                y_hat = self.predict(X)
                cost = mse_loss(y_hat, y)

                if self.reg == "L2":
                    cost += self.l2_reg()

            if self.reg:
                losses.append(cost[0, 0].numpy())
            else:
                losses.append(cost.numpy())

            if self.val:
                y_val = self.predict(val_X)
                val_loss = mse_loss(y_val, val_y)
                val_losses.append(val_loss.numpy())

            gradients = g.gradient(cost, [self.bias, self.weights])
            self.optimizer.apply_gradients(zip(gradients, [self.bias, self.weights]))

            if self.verbosity:
                print("Epoch %d, loss: %.3f" % ((i + 1), cost.numpy()))
        return losses, val_losses
