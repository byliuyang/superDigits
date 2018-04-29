import numpy 

class NN():
    def __init__(self, learning_rate, regularization_strength, hidden_units):
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.l2_strength = regularization_strength

    def f_ce(y_hat, y):
        return -1 * np.sum(y * np.log(y_hat))

    def f_pc(y_hat, y):
        pc_correct = np.mean(np.argmax(y_hat, axis=1) == np.argmax(y, axis=1))
        return pc_correct

    def fit(self, x, y):
        #hyper parameters

        tolerance = .0000001
        batch_size = 60
        target_epochs = 35

        #shuffle the dataset and Transpose X
        x, y = shuffleDataset(x, y)
        W = .01 * np.random.rand(X.shape[0], y.shape[1])

        is_training = True
        current_epochs = 0

        self.w_1 = rand_weights(self.hidden_units, x.rows)
        self.w_2 = rand_weights(y.rows, self.hidden_units)
        self.b_1 = .01 # TODO vectorize
        self.b_2 = .01 # TODO vectorize

        while is_training and current_epochs < target_epochs:
            for i in xrange(0, x.shape[1], batch_size):
                x_batch = x[:,i:i+batch_size]
                y_batch = y[i:i+batch_size]

                y_hat, h_1, z_1 = forward_prop(x_batch)
                grad_w1, grad_w2, grad_b1, grad_b2 = back_prop(h_1, z_1, x_batch, y_batch, y_hat)

                w_1 -= self.learning_rate * grad_w1 
                w_2 -= self.learning_rate * grad_w2
                b_1 -= self.learning_rate * grad_b1
                b_2 -= self.learning_rate * grad_b2
            
            current_epochs+=1
            epoch_yhat = yhat_softmax(X, W)
            epoch_cost = f_ce(epoch_yhat, y)
            epoch_accuracy = f_pc(epoch_yhat, y) * 100

            if abs(epoch_cost) < tolerance:
                    is_training = False

            print("Epoch {0}: {1:.2f}% accuracy, {2} cost".format(current_epochs, epoch_accuracy, epoch_cost))
        
        def predict(faces):
            return forward_prop(x_batch)[0] # first element in tuple is y_hat

        Classifier =  namedtuple("nn", ["W", "predict"])
        return Classifier(W, predict)

    def rand_weights(rows, cols):
        return np.random.randn(rows, cols)

    def forward_prop(x):
        # requires
        z_1 = self.w_1 * x + self.b_1
        h_1 = relu(z_1)
        z_2 = self.w_2 * h_1 + self.b_2
        y_hat = softmax(z_2)

     return y_hat, h_1, z_1

    def relu(input):
        # TODO implement relu

    def backward_prop(h_1, z_1, x, y, y_hat):
        g = ((y - y_hat).T.dot(w_2) * relu_prime(z1.T)).T

        grad_w1 = g * x.T + self.l2_strength * self.w_1
        grad_w2 = (y_hat - y) * h_1.T + self.l2_strength * self.w_2
        grad_b1 = g
        grad_b2 = y_hat - y

        return grad_w1, grad_w2, grad_b1, grad_b2

def main():
    learning_rate = .5
    hidden_units = 5
    regularization_strength = .01

    trainImages, trainLabels = loadData("training")

    nn_classifier = NN(learning_rate, hidden_units)
    nn_classifier.fit(trainImages, trainLabels)

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    main()
