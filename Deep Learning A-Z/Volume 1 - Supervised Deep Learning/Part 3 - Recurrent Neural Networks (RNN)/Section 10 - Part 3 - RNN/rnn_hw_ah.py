# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

total_set = pd.read_csv('GOOG_Current.csv')
ti = int(len(total_set) * 0.75)
training_set = total_set.iloc[0:ti,1:2].values
test_set = total_set.iloc[ti:,1:2].values


sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)


X_train = training_set[0:-1]
y_train = training_set[1:]

X_train = np.reshape(X_train, (len(X_train),1,1))

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

model = Sequential()
model.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=200)


# testing
real_price = test_set[:,:]
test_set = sc.transform(test_set)
test_set = np.reshape(test_set, (len(test_set),1,1))
predicted = model.predict(test_set)
predicted = sc.inverse_transform(predicted)

import matplotlib.pyplot as plt
plt.plot(real_price, color = 'red', label = "real")
plt.plot(predicted, color = 'blue', label = "predicted")
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()



