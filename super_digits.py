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
        pass
        # m = training_images.shape[1]
        # c = training_labels.shape[1]
        self._W_1 = 0.001 * np.random.rand(m, c)
        self._W_2 = 0.001 * np.random.rand(m, c)
        self._b_1 = 0.001 * np.random.rand(m, c)
        self._b_2 = 0.001 * np.random.rand(m, c)

        # indices = np.arange(len(training_images))
        # np.random.shuffle(indices)

        # training_images = training_images[indices, :]
        # training_labels = training_labels[indices, :]

        # training_set_size = training_images.shape[0]
        # learning_rate = 0.8
        # anneal_rate = 15000
        # decay_rate = 0.95
        # rounds = 0

        for epoch in range(self._epochs):
            for batch_start in range(0, training_set_size, self._batch_size):
                batch_end = batch_start + batch_size
                X = training_images[batch_start:batch_end].T
                Y = training_labels[batch_start:batch_end]
                # Compute Gradients

                x = 
                y = 

                z_1, h1, y_hat = self._forward_propagation(x)
                self._backward_propagation(x, y, z_1, h_1, y_hat)
            
                rounds += 1
                if rounds % anneal_rate == 0:
                    learning_rate *= decay_rate
            
            print("Epoch %3d/%3d  Loss = %.2f" % (epoch + 1, epochs, cross_entropy_loss(Y, yhat(X, W))))
        
    def _forward_propagation(self, x):
        z_1 = self._W_1.dot(x) + self._b_1
        h_1 = self._relu(z_1)
        z_2 = self._W_2.dot(h_1) + self._b_2
        y_hat = self._softmax(z_2)
        return z_1, h_1, y_hat
    
    def _backward_propagation(self, x, y, z_1, h_1, y_hat):
        df_dy = y_hat - y
        g = self._g(self, df_dy, self._W_2, z_1) 

        self._W_1 = self._W_1 - self._learning_rate * self._W_1_prime(x, g, self._W_1, self._l_2_alpha_1, self._l_1_beta_1)
        self._W_2 = self._W_2 - self._learning_rate * self._W_2_prime(df_dy, h_1, self._W_2, self._l_2_alpha_2, self._l_1_beta_2)

        self._b_1 = self._b_1 - self._learning_rate * self._b_1_prime(g)
        self._b_2 = self._b_2 - self._learning_rate * self._b_2_prime(df_dy)

    def predict(self, X):
        pass
    
    def _relu(self, x):
        return 0 if x <= 0 else x

    def _relu_prime(self, x):
        return 0 if x <= 0 else 1

    def _z_1(self, x, W_1, b_1):
        return W_1.T.dot(x) + b_1
    
    def _h_1(self, z_1):


    def _yhat(X, W):
        Z = W.T.dot(X)
        return softmax(Z).T

    def _softmax(Z):
        exp = np.exp(Z)
        total = np.sum(exp, axis=0)
        return exp / total
    
    def _g(self, df_dy, W_2, z_1):
        return (df_dy.T.dot(W_2) * self._relu_prime(z_1.T)).T

    def _W_2_prime(self, df_dy, h_1, W_2, alpha_2, beta_2):
        return df_dy.dot(h_1.T) + alpha_2 * W_2 + beta_2 * np.sign(W_2) 
    
    def _b2_prime(self, df_dy):
        return df_dy

    def _W_1_prime(self, x, g, W_1, alpha_1, beta_1):
        return g.dot(x.T) + alpha_1 * W_1 + beta_1 * np.sign(W_1) 

    def _b1_prime(self, g):
        return g

    def _cross_entropy_loss(y, yhat):
        loss = 0
        yhat_log = np.log(yhat.T)
        for i in range(len(y)):
            loss -= y[i, :].dot(yhat_log[:, i]) 
        return loss

    def _toClassIndices(self, probabilities):
        return np.argmax(probabilities, axis=1)

    def loss(self, testing_labels, predicted_labels):
        return 0

    def score(self, expected_labels, predicted_labels):
        return np.mean(self._toClassIndices(expected_labels) == self._toClassIndices(predicted_labels))

def recognize_digit(training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels):
    print("Start training...")
    print()

    clf = NeuralNetworkClassifier(hidden_units=30, learning_rate=0.001, batch_size=16, epochs=30, l1_beta=0.5, l2_alpha=0.4)
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

    recognize_digit(training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels)

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python3 digit_recognizer.py")
        exit()
    main()