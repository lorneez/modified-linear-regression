import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from modified_lr import ModifiedLogisticRegression
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# data = load_breast_cancer()
# X, y = data.data, data.target

lc_data = pd.read_csv("modified-linear-regression/data_2.csv")
lc_data = shuffle(lc_data)
X = lc_data[['term','int_rate','loan_amnt','annual_inc','installment','dti','verification_status']]
cols_to_norm = ['term','int_rate','loan_amnt','annual_inc','installment','dti','verification_status']
X[cols_to_norm] = X[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# with pd.option_context('display.max_rows', 5, 'display.max_columns', None):  # more options can be specified also
#   print(X.head)
X = X.to_numpy()
y = lc_data[['loan_status']].to_numpy()
y = np.reshape(y,(y.shape[0],))


print(X.shape)
print(y.shape)

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


# percent of positives
def numberRecommended(y_hat):
    total = 0
    recommended = 0
    for i in range(y_hat.shape[0]):
        total += 1
        recommended += y_hat[i]
    return recommended / total


def testMLR():
    for b in range(10, 30):
        beta = b / 10
        MLR = ModifiedLogisticRegression(learning_rate=0.1, num_iterations=1000, beta=beta)
        MLR.fit(X_train, y_train)
        predictions = MLR.predict(X_test)
        # MLR.save()
        print("BETA:", beta)
        print("Accuracy:", getAccuracy(y_test, predictions))
        print("Precision:", getPrecision(y_test, predictions))
        print("Number Recommended:", numberRecommended(predictions))
        print("")


testMLR()
