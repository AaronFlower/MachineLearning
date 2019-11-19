import numpy as np
import tensorflow as tf


class MulticlassLinear(object):

    def __init__(self, epochs=1000, learning_rate=0.01, verbosity=1):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbosity = verbosity
        self.bias = None
        self.weights = None
        self.optimizer = tf.optimizers.Adagrad(learning_rate)

    def init_weights(self, n_features, n_classes):
        self.bias = tf.Variable(np.zeros(n_classes), dtype=tf.float64,  name="bias")
        # self.weights = tf.Variable(np.random.randn(n_features, n_classes), dtype=tf.float64, name="weights")
        self.weights = tf.Variable(np.zeros((n_features, n_classes)), dtype=tf.float64, name="weights")
        if self.verbosity:
            print("Init Bias:", self.bias)
            print("Init Weighs: ", self.weights)

    def predict(self, X):
        y_score = tf.nn.softmax(tf.matmul(X, self.weights) + self.bias)
        return tf.argmax(y_score, axis=1)

    def train(self, dataset, X_val=None, y_val=None):
        losses = []
        losses_val = []
        dataset = dataset.take(self.epochs)
        for (i, (X, y)) in enumerate(dataset):
            if self.weights is None:
                self.init_weights(X.shape[1], y.shape[1])

            with tf.GradientTape() as g:
                y_pred = tf.nn.softmax(tf.matmul(X, self.weights) + self.bias)
                loss = tf.reduce_mean(-y * tf.math.log(y_pred))
            gradients = g.gradient(loss, [self.weights, self.bias])
            self.optimizer.apply_gradients(zip(gradients, [self.weights, self.bias]))
            losses.append(loss.numpy())
            if self.verbosity:
                print("loss: %g" % loss)

            if X_val is not None:
                y_val_pred = tf.nn.softmax(tf.matmul(X_val, self.weights) + self.bias)
                loss_val = tf.reduce_mean(-y_val * tf.math.log(y_val_pred))
                losses_val.append(loss_val.numpy())

        return losses, losses_val
