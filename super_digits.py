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
        self._W_2 = 1 / np.sqrt(self._num_classes) * np.random.randn(self._num_classes, self._hidden_units)
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
            y_hat = self.predict(X_train)
            print("Epoch %3d/%3d  Loss = %.2f Training Accuracy = %.2f" % (epoch + 1, self._epochs,self._cross_entropy_loss(Y_batch, Y_hats), self.score(Y_train, y_hat)))

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

def describe_hyperparameters(hyperparameters):
     return "\nHidden Units: {0} Learning Rate: {1} Minibatch Size: {2} Epochs: {3} L1 Strength: {4} L2 Strength: {5}".format(
                hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3], hyperparameters[4], hyperparameters[5])


def findBestHyperparameters(training_images, training_labels, validation_images, validation_labels):
    print("Start training...")
    print()

    all_hidden_units = [20, 20, 30, 30, 40, 40, 50, 50, 60, 30]
    all_learning_rates = [0.0001, 0.001, 0.01, 0.01, 0.01, 0.02, 0.02, 0.1, 0.2, 0.007]
    all_minibatch_sizes = [2, 5, 10, 10, 20, 20, 100, 50, 50, 25]
    all_num_epochs = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3]
    all_l1_strengths = [0.0, 0.0, 0, 0.01, 0.0, 0.001, 0.01, 0.02, 0.01, 0.001]
    all_l2_strengths = [0.0, 0.01, 0.001, 0.0, 0.01, 0.001, 0.01, 0.02, 0.01, 0.001]

    slice_start = 0
    slice_size = 5

    best_accuracy = 0
    best_hyperparamters = [] 

    for i in range(slice_size):
        hyperparameters = (all_hidden_units[slice_start+i], 
                            all_learning_rates[slice_start+i],
                            all_minibatch_sizes[slice_start+i],
                            all_num_epochs[slice_start+i],
                            all_l1_strengths[slice_start+i],
                            all_l2_strengths[slice_start+i])

        print(describe_hyperparameters(hyperparameters))

        clf = NeuralNetworkClassifier(
                        hidden_units = hyperparameters[0],
                        learning_rate = hyperparameters[1], 
                        batch_size = hyperparameters[2], 
                        epochs = hyperparameters[3], 
                        l_1_beta_1 = hyperparameters[4], 
                        l_1_beta_2 = hyperparameters[4], 
                        l_2_alpha_1 = hyperparameters[5], 
                        l_2_alpha_2 = hyperparameters[5])

        clf.fit(training_images, training_labels)

        predicted_labels = clf.predict(validation_images)

        accuracy = clf.score(validation_labels, predicted_labels)

        print("Accuracy: %f" % accuracy)
        print("Cross Entropy Loss = %.2f" % (clf.loss(validation_labels, predicted_labels)))

        if(accuracy > best_accuracy):
            best_accuracy = accuracy
            best_hyperparamters = hyperparameters
            print("Found new best hyperparameters.")
        
        print("\n")
    
    print(describe_hyperparameters(best_hyperparamters))
    return best_hyperparamters

def main():
    training_images = np.load("mnist_train_images.npy")
    training_labels = np.load("mnist_train_labels.npy")
    testing_images = np.load("mnist_test_images.npy")
    testing_labels = np.load("mnist_test_labels.npy")

    #TODO replace with validation set
    validation_images = testing_images
    validation_labels = testing_labels
 
    parameters = findBestHyperparameters(training_images[0:16000, :], training_labels[0:16000, :], 
                        validation_images, validation_labels)

    clf = NeuralNetworkClassifier(hidden_units=parameters[0], 
    learning_rate=parameters[1], 
    batch_size=parameters[2], 
    epochs=parameters[3], l_1_beta_1=parameters[4], l_1_beta_2=parameters[4], l_2_alpha_1=parameters[5], l_2_alpha_2=parameters[5])

    clf.fit(training_images, training_labels)
    predicted_labels = clf.predict(testing_images)

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Usage: python3 digit_recognizer.py")
        exit()
    main()