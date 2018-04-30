import numpy as np
import sys
from collections import namedtuple
from matplotlib import pyplot as plt


class NeuralNetworkClassifier():
    def __init__(self, hidden_units, learning_rate, batch_size, epochs, l_1_beta_1, l_1_beta_2, l_2_alpha_1, l_2_alpha_2):
        self._hidden_units = hidden_units
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._epochs = epochs
        self._l_1_beta_1 = l_1_beta_1
        self._l_1_beta_2 = l_1_beta_2
        self._l_2_alpha_1 = l_2_alpha_1
        self._l_2_alpha_2 = l_2_alpha_2

    def fit(self, X_train, Y_train):
        num_input_dimensions = X_train.shape[1]
        self._num_classes = Y_train.shape[1]
        training_set_size = X_train.shape[0]

        self._W_1 = 1 / np.sqrt(self._hidden_units) * np.random.randn(self._hidden_units, num_input_dimensions)
        self._W_2 = 1 / np.sqrt(self._hidden_units) * np.random.randn(self._num_classes, self._hidden_units)
        self._b_1 = 0.01 * np.ones((self._hidden_units, 1))
        self._b_2 = 0.01 * np.ones((self._num_classes, 1))

        for epoch in range(self._epochs):
            for batch_start in range(0, training_set_size, self._batch_size):
                batch_end = batch_start + self._batch_size
                X_batch = X_train[batch_start:batch_end]
                Y_batch = Y_train[batch_start:batch_end]

                num_examples = X_batch.shape[0]

                W_1_prime_total = 0
                W_2_prime_total = 0
                b_1_prime_total = 0
                b_2_prime_total = 0

                for i in range(num_examples):
                    x = np.vstack(X_batch[i, :])
                    y = np.vstack(Y_batch[i, :])

                    z_1, h_1, y_hat = self._forward_propagation(x)
                    W_1_prime, W_2_prime, b_1_prime, b_2_prime = self._backward_propagation(x, y, z_1, h_1, y_hat)

                    W_1_prime_total += W_1_prime
                    W_2_prime_total += W_2_prime
                    b_1_prime_total += b_1_prime
                    b_2_prime_total += b_2_prime
            
                self._W_1 = self._W_1 - self._learning_rate * W_1_prime_total
                self._W_2 = self._W_2 - self._learning_rate * W_2_prime_total
                self._b_1 = self._b_1 - self._learning_rate * b_1_prime_total
                self._b_2 = self._b_2 - self._learning_rate * b_2_prime_total
            
            Y_hats = self.predict(X_batch)
            print("Epoch %3d/%3d  Loss = %.2f" % (epoch + 1, self._epochs,self._cross_entropy_loss(Y_batch, Y_hats)))

    def _forward_propagation(self, x):
        z_1 = self._W_1.dot(x) + self._b_1

        # print("_forward_propagation W_1=", self._W_1.shape)
        # print("_forward_propagation b_1=", self._b_1.shape)
        # print("_forward_propagation x=", x.shape)
        # print("_forward_propagation z=", z_1.shape)
        h_1 = self._relu(z_1)

        # print("_forward_propagation h_1=", h_1.shape)
        z_2 = self._W_2.dot(h_1) + self._b_2

        # print("_forward_propagation z_2=", z_2.shape)
        y_hat = self._softmax(z_2)

        # print("_forward_propagation y_hat=", y_hat.shape)
        return z_1, h_1, y_hat
    
    def _backward_propagation(self, x, y, z_1, h_1, y_hat):
        df_dy = y_hat - y
        g = self._g(df_dy, self._W_2, z_1)

        W_1_prime = self._W_1_prime(x, g, self._W_1, self._l_2_alpha_1, self._l_1_beta_1)
        W_2_prime = self._W_2_prime(df_dy, h_1, self._W_2, self._l_2_alpha_2, self._l_1_beta_2)
        b_1_prime = self._learning_rate * self._b_1_prime(g)
        b_2_prime = self._learning_rate * self._b_2_prime(df_dy)

        return W_1_prime, W_2_prime, b_1_prime, b_2_prime

    def predict(self, X):
        num_examples = X.shape[0]
        Y_hat = np.zeros((num_examples, self._num_classes))
        for i in range(num_examples):
            x = np.vstack(X[i, :])
            _, _, y_hat = self._forward_propagation(x)
            Y_hat[i, :] = y_hat[:, 0]
        return Y_hat
    
    def _relu(self, x):
        return np.maximum(x, 0)

    def _relu_prime(self, x):
        y = np.zeros((x.shape[0], x.shape[1]))
        y[x > 0] = 1.0
        return y

    def _softmax(self, Z):
        exp = np.exp(Z)
        total = np.sum(exp, axis=0)
        return exp / total
    
    def _g(self, df_dy, W_2, z_1):
        return (df_dy.T.dot(W_2) * self._relu_prime(z_1.T)).T

    def _W_2_prime(self, df_dy, h_1, W_2, alpha_2, beta_2):
        return df_dy.dot(h_1.T) + alpha_2 * W_2 + beta_2 * np.sign(W_2) 
    
    def _b_2_prime(self, df_dy):
        return df_dy

    def _W_1_prime(self, x, g, W_1, alpha_1, beta_1):
        return g.dot(x.T) + alpha_1 * W_1 + beta_1 * np.sign(W_1) 

    def _b_1_prime(self, g):
        return g

    def _l_1_loss(self, W):
        return np.sum(np.absolute(W))
    
    def _l_2_loss(self, W):
        return 0.5 * np.linalg.norm(W)

    def _cross_entropy_loss(self, y, yhat):
        loss = 0
        yhat_log = np.log(yhat.T)
        for i in range(len(y)):
            loss -= y[i, :].dot(yhat_log[:, i])

        l_1_regularization = self._l_1_beta_1 * self._l_1_loss(self._W_1) + self._l_1_beta_2 * self._l_1_loss(self._W_2)
        l_2_regularization = self._l_2_alpha_1 * self._l_2_loss(self._W_1) + self._l_2_alpha_2 * self._l_2_loss(self._W_2)
        return loss + l_1_regularization + l_2_regularization

    def _toClassIndices(self, probabilities):
        return np.argmax(probabilities, axis=1)

    def loss(self, testing_labels, predicted_labels):
        return 0

    def score(self, expected_labels, predicted_labels):
        return np.mean(self._toClassIndices(expected_labels) == self._toClassIndices(predicted_labels))

def recognize_digit(training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels):
    print("Start training...")
    print()

    clf = NeuralNetworkClassifier(hidden_units=30, learning_rate=0.001, batch_size=16, epochs=30, l_1_beta_1=0.6, l_1_beta_2=0.5, l_2_alpha_1=0.4, l_2_alpha_2=0.3)
    clf.fit(training_images, training_labels)
    predicted_labels = clf.predict(testing_images)

    print()
    print("Cross Entropy Loss = %.2f" % (clf.loss(testing_labels, predicted_labels)))
    print("Accuracy: %f" % clf.score(testing_labels, predicted_labels))

def main():
    training_images = np.load("mnist_train_images.npy")
    training_labels = np.load("mnist_train_labels.npy")
    validation_images = np.load("mnist_validation_images.npy")
    validation_labels = np.load("mnist_validation_labels.npy")
    testing_images = np.load("mnist_test_images.npy")
    testing_labels = np.load("mnist_test_labels.npy")

    recognize_digit(training_images[0:160, :], training_labels[0:160, :], validation_images, validation_labels, testing_images[0:160, :], testing_labels[0:160, :])

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python3 digit_recognizer.py")
        exit()
    main()