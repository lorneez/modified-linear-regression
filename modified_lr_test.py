import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from modified_lr import ModifiedLogisticRegression

# load data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# accuracy
def getAccuracy(y_true, y_hat):
    accuracy = np.sum(y_true == y_hat) / len(y_true)
    return accuracy


# precision
def getPrecision(y_true, y_hat):
    true_positive = 0
    total = 0
    for i in range(y_hat.shape[0]):
        if y_hat[i] == 1:
            total += 1
            true_positive += y_true[i]
    if total > 0:
        return true_positive / total
    else:
        return 0

def numberRecommended(y_hat):
    total = 0
    recommended = 0
    for i in range(y_hat.shape[0]):
        total += 1
        recommended += y_hat[i]
    return recommended / total


def testMLR():
    for b in range(10, 40):
        beta = b/10
        MLR = ModifiedLogisticRegression(learning_rate=0.0001, num_iterations=1000, beta=beta)
        MLR.fit(X_train, y_train)
        predictions = MLR.predict(X_test)
        print("BETA:", beta)
        print("Accuracy:", getAccuracy(y_test, predictions))
        print("Precision:", getPrecision(y_test, predictions))
        print("Number Recommended:", numberRecommended(predictions))
        print("")

testMLR()
