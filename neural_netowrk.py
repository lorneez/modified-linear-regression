from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lc_data = pd.read_csv("/Users/lorneez/PycharmProjects/pythonProject3/modified-linear-regression/data_2.csv")
# lc_data = shuffle(lc_data)
X = lc_data[['term','int_rate','loan_amnt','annual_inc','installment','dti','verification_status']]
cols_to_norm = ['term','int_rate','loan_amnt','annual_inc','installment','dti','verification_status']
X[cols_to_norm] = X[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# with pd.option_context('display.max_rows', 5, 'display.max_columns', None):  # more options can be specified also
X = X.to_numpy()
print(X)
y = lc_data[['loan_status']].to_numpy()
y = np.reshape(y,(y.shape[0],))


print(X.shape)
print(y.shape)
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = Sequential()
model.add(Dense(1000, input_dim=7, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_test, y_test))

model.save('model_1')

_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))