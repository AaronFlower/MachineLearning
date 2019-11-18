import numpy as np
import tensorflow as tf


def cross_entropy_loss(y_pred, y):
    """
    cross_entropy loss 一定要写对呀
    """
    cost = tf.reduce_mean(-y * tf.math.log(y_pred) - (1 - y) * tf.math.log(1 - y_pred))
    return cost


class LRModel(object):

    def __init__(self, epochs=100, optimizer=None, verbosity=1):
        self.name = "Logistic Model"
        self.epochs = epochs
        self.verbosity = verbosity

        self.bias = None
        self.weights = None

        if optimizer:
            self.optimizer = optimizer
        else:
            # self.optimizer = tf.optimizers.Adagrad(0.01)
            self.optimizer = tf.optimizers.SGD(0.5)

    def init_weight(self, n_features):
        self.bias = tf.Variable(np.zeros(1), dtype=tf.float64, name="bias")
        self.weights = tf.Variable(np.random.randn(n_features, 1), dtype=tf.float64, name="weights")
        if self.verbosity:
            print("Init Weights: ", self.weights)
            print("Init Bias: ", self.bias)

    def predict(self, X):
        y_hat = self.sigmoid(X)
        y_pred = np.zeros((len(y_hat), 1))
        y_pred[y_hat >= 0.5] = 1
        return y_pred

    def sigmoid(self, X):
        z = self.bias + tf.matmul(X, self.weights)
        return tf.sigmoid(z)

    def train(self, dataset, X_val=None, y_val=None):
        losses = []
        val_losses = []
        dataset = dataset.take(self.epochs)
        for i, data_shot in enumerate(dataset):
            if self.weights is None:
                self.init_weight(data_shot[0].shape[1])
            with tf.GradientTape() as g:
                X, y = data_shot
                y_hat = self.sigmoid(X)
                loss = cross_entropy_loss(y_hat, y)

            gradients = g.gradient(loss, [self.bias, self.weights])

            self.optimizer.apply_gradients(zip(gradients, [self.bias, self.weights]))

            if self.verbosity:
                print("Epoch %d, loss: %g" % (i + 1, loss.numpy()))
            losses.append(loss)
            if X_val is not None:
                y_val_hat = self.sigmoid(X_val)
                val_loss = cross_entropy_loss(y_val_hat, y_val)
                val_losses.append(val_loss)
        return losses, val_losses
