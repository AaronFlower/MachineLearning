import numpy as np
import tensorflow as tf


def cross_entropy_loss(y_pred, y):
    cost = tf.reduce_mean(-y * tf.math.log(y_pred))
    return cost


class LRModel(object):

    def __init__(self,
                 dataset,
                 epochs=100,
                 optimizer=None,
                 verbosity=1):
        self.name = "Logistic Model"
        self.dataset = dataset
        self.epochs = epochs
        self.verbosity = verbosity

        data_shot = dataset.take(1)
        for X, _ in data_shot:
            _, self.n_features = X.shape

        self.bias = tf.Variable(np.zeros(1), dtype=tf.float64, name="bias")
        self.weights = tf.Variable(np.random.randn(self.n_features, 1))

        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = tf.optimizers.Adagrad(0.1)

        if self.verbosity:
            print("Init Weights: ", self.weights)
            print("Init Bias: ", self.bias)

    def predict(self, X, trained=False):
        h = self.bias + tf.matmul(X, self.weights)
        y_hat = tf.sigmoid(h)
        # if trained:
        return y_hat

        # y_hat[y_hat >= 0.5] = 1
        # y_hat[y_hat < 0.5] = 0
        # return y_hat

    def train(self):
        dataset = self.dataset.take(self.epochs)
        for i, data_shot in enumerate(dataset):
            with tf.GradientTape() as g:
                X, y = data_shot
                y_hat = self.predict(X, trained=True)
                loss = cross_entropy_loss(y_hat, y)

            gradients = g.gradient(loss, [self.bias, self.weights])

            self.optimizer.apply_gradients(zip(gradients, [self.bias, self.weights]))

            if self.verbosity:
                print("Epoch %d, loss: %.4f" %(i + 1, loss.numpy()))
