import numpy as np
import sys
from collections import namedtuple
from matplotlib import pyplot as plt

def yhat(X, W):
    Z = W.T.dot(X)
    return softmax(Z).T

def softmax(Z):
    exp = np.exp(Z)
    total = np.sum(exp, axis=0)
    return exp / total

def gradient(X, Y, W):
    n = X.shape[1]
    return X.dot(yhat(X, W) - Y) / n

def cross_entropy_loss(y, yhat):
    loss = 0
    yhat_log = np.log(yhat.T)
    for i in range(len(y)):
        loss -= y[i, :].dot(yhat_log[:, i]) 
    return loss

def toClassIndices(probabilities):
    return np.argmax(probabilities, axis=1)

def accuracy(expected_labels, predicted_labels):
    return np.mean(toClassIndices(expected_labels) == toClassIndices(predicted_labels))

def digit_classifier(training_images, training_labels, epochs=100, batch_size=100):
    m = training_images.shape[1]
    c = training_labels.shape[1]
    W = 0.001 * np.random.rand(m, c)

    indices = np.arange(len(training_images))
    np.random.shuffle(indices)

    training_images = training_images[indices, :]
    training_labels = training_labels[indices, :]

    training_set_size = training_images.shape[0]
    learning_rate = 0.8
    anneal_rate = 15000
    decay_rate = 0.95
    rounds = 0

    for epoch in range(epochs):
        for batch_start in range(0, training_set_size, batch_size):
            batch_end = batch_start + batch_size
            X = training_images[batch_start:batch_end].T
            Y = training_labels[batch_start:batch_end]
            W = W - learning_rate * gradient(X, Y, W)
            rounds += 1
            if rounds % anneal_rate == 0:
                learning_rate *= decay_rate
        
        print("Epoch %3d/%3d  Loss = %.2f" % (epoch + 1, epochs, cross_entropy_loss(Y, yhat(X, W))))
    
    def predict(images):
        return yhat(images.T, W)
    
    Classifier = namedtuple("Classifier", ["W", "predict"])
    return Classifier(W, predict)

def visualize(images, labels, rows = 5, cols = 20):
    n = rows * cols
    images = images[:n]
    labels = labels[:n, :]

    class_indices = toClassIndices(labels)
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    images = np.reshape(images, (images.shape[0], 28, 28))

    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.title(class_names[class_indices[i]])
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')
        
    plt.show()

def recognize_digit(training_images, training_labels, testing_images, testing_labels):
    print("Start training...")
    print()
    
    epochs = int(sys.argv[1])
    clf = digit_classifier(training_images, training_labels, epochs=epochs)
    predicted_labels = clf.predict(testing_images)

    print()
    print("Cross Entropy Loss = %.2f" % (cross_entropy_loss(testing_labels, predicted_labels)))
    print("Accuracy: %f" % accuracy(testing_labels, predicted_labels))

    visualize(testing_images, predicted_labels)

def main():
    training_images = np.load("mnist_train_images.npy")
    training_labels = np.load("mnist_train_labels.npy")
    testing_images = np.load("mnist_test_images.npy")
    testing_labels = np.load("mnist_test_labels.npy")

    recognize_digit(training_images, training_labels, testing_images, testing_labels)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 digit_recognizer.py [number of epochs]")
        print("Eg: python3 digit_recognizer.py 20")
        exit()
    main()