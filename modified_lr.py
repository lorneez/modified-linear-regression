import numpy as np
import scipy
from scipy.special import expit

# sigmoid function
def sigmoid(x):
    # return 1 / (1 + scipy.special.expit(-x))
    return 1 / (1 + np.exp(-x))

# modified logistic regression model used for high precision classifying
class ModifiedLogisticRegression:

    # constructor:
    # learning rate = 0.001
    # number of iterations = 1000
    # beta = 1 (default logistic regression)
    def __init__(self, learning_rate=0.001, num_iterations=1000, beta=1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.beta = beta  # beta represents a weight that we put on misclassified positives

    # fit the model
    def fit(self, X, y):
        # init parameters
        num_samples, num_features = X.shape
        self.theta = np.zeros(num_features + 1)
        X_mod = np.zeros(shape=(num_samples, num_features + 1))

        # add 1 to end of data for bias
        for i in range(num_samples):
            X_mod[i] = np.append(X[i], [1])

        # gradient descent
        batch_size = 128
        for i in range(self.num_iterations):

            indices = np.random.choice(range(len(X_mod)), batch_size)
            batch_data = X_mod[indices]
            batch_labels = y[indices]
            if i % 10 == 0:
                # get predictions
                linear_model = np.dot(batch_data, self.theta)
                y_hat = sigmoid(linear_model)

                loss = 0
                for i in range(batch_size):
                    loss += -(batch_labels[i] * np.log(y_hat[i])) - (1 - batch_labels[i]) * self.beta * np.log(1 - y_hat[i])
                print(loss / batch_size)

            # calculate the derivative of the modified logistic regression function
            xw = np.dot(batch_data, self.theta)
            # print("xw",xw)
            # print("theta",self.theta)
            exw = np.exp(xw)
            # exw = scipy.special.expit(xw)
            a = self.beta * batch_labels - self.beta
            # print("a",a)
            b = a * exw + batch_labels
            # print("b",b)
            c = - b / (exw + 1)
            # print("c",c)
            gradient = np.dot(batch_data.T, c) * (1 / batch_size)
            # print("gradient",gradient)
            # update function
            # print("learning rate", self.learning_rate)
            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        print(self.theta)
        # init parameters
        num_samples, num_features = X.shape
        X_mod = np.zeros(shape=(num_samples, num_features + 1))

        # add 1 to end of data for bias
        for i in range(num_samples):
            X_mod[i] = np.append(X[i], [1])

        # get predictions
        linear_model = np.dot(X_mod, self.theta)
        y_hat = sigmoid(linear_model)
        y_classify = np.zeros(num_samples)

        # classify
        for i in range(len(y_hat)):
            if y_hat[i] > 0.5:
                y_classify[i] = 1
            else:
                y_classify[i] = 0
        return y_classify

    # def save(self, X):

