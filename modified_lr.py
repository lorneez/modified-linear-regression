import numpy as np


class ModifiedLogisticRegression:

    def __init__(self, learning_rate=0.001, num_iterations=1000, beta=1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.beta = beta

    def fit(self, X, y):
        # init parameters
        num_samples, num_features = X.shape
        # self.theta = np.random.rand(num_features + 1)
        self.theta = np.zeros(num_features + 1)
        X_mod = np.zeros(shape=(num_samples, num_features + 1))

        # add 1 to end of data for bias
        for i in range(num_samples):
            X_mod[i] = np.append(X[i], [1])

        # gradient descent
        for i in range(self.num_iterations):
            # linear_model = np.dot(X_mod, self.theta)
            # y_hat = self.sigmoid(linear_model)
            # gradient = (1 / num_samples) * np.dot(X_mod.T, y_hat - y)

            xw = np.dot(X_mod, self.theta)
            exw = np.exp(xw)
            a = self.beta * y - self.beta
            b = a + y / exw
            c = b / (1 + 1/exw)
            gradient = -np.dot(X_mod.T, c) * (1/num_samples)

            self.theta -= self.learning_rate * gradient

    def predict(self, X):
        # init parameters
        num_samples, num_features = X.shape
        X_mod = np.zeros(shape=(num_samples, num_features + 1))

        # add 1 to end of data for bias
        for i in range(num_samples):
            X_mod[i] = np.append(X[i], [1])

        linear_model = np.dot(X_mod, self.theta)
        y_hat = self.sigmoid(linear_model)
        y_classify = np.zeros(num_samples)
        for i in range(len(y_hat)):
            if y_hat[i] > 0.5:
                y_classify[i] = 1
            else:
                y_classify[i] = 0
        return y_classify

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
