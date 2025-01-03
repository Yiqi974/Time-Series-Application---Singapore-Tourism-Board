# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:25:00 2023

@author: Brynn
"""
import pandas as pd
import numpy as np
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential, load_model
from keras import optimizers
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from scipy import stats
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import os

# load the dataset
cwd = os.getcwd()
os.chdir(cwd)
df = pd.read_csv('cleaned.csv')

# define the data
x = df['Total']
t = df['Data Series']

# split the data into training and test sets
train_size = int(len(x) * 0.7)
test_size = len(x) - train_size
train_x, test_x = x[0:train_size], x[train_size:len(x)]
max_train = max(test_x)
min_train = min(test_x)
t_plot = t[train_size+3:len(x)]
test_x_plot = test_x[3:]
train_x_plot = train_x

# normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
train_x = scaler.fit_transform(train_x.values.reshape(-1, 1))
test_x = scaler.fit_transform(test_x.values.reshape(-1, 1))

# Create a sliding window of the training data for use as input to the LSTM model
train_generator = TimeseriesGenerator(train_x, train_x, length=3, batch_size=128)

# Create a sliding window of the test data for use as input to the LSTM model
test_generator = TimeseriesGenerator(test_x, test_x, length=3, batch_size=1)

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(3, 1)))
# model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Dense(units=100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))

adam = optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=adam, loss='mean_squared_error')

# Fit the model to the training data
model.fit(train_generator, epochs=50)
# model.save('model.h5')
# model = load_model('model.h5')
# Evaluate the model on the test data
test_predictions = model.predict(test_generator)
rmse = np.sqrt(mean_squared_error(test_x[3:], test_predictions))
print('Root Mean Squared Error:', rmse)

xreverse = test_predictions * (max_train - min_train) + min_train
xreverse = xreverse.flatten()

# future prediction
# Scale the input sequence
scaler = MinMaxScaler(feature_range=(0, 1))
lastThree = scaler.fit_transform(x.values.reshape(-1, 1))[-4:-1]

# Reshape the input sequence
lastThree = lastThree.reshape(1, lastThree.shape[0], 1)
# Store the predicted values
predicted = []

# Predict the next 3 values
for i in range(36):
    # Use the model to predict the next value
    yhat = model.predict(lastThree)
    predicted.append(yhat[0,0])
    
    # Update the input sequence
    lastThree = np.append(lastThree[:,1:,:], [[yhat[0]]], axis=1)
future = np.array(predicted) * (max_train - min_train) + min_train

df = pd.read_csv('date.csv')
date = df['date']
x_predict = t_plot.values.tolist()+date.values.tolist()
y_predict = [np.nan]*159+future.tolist()

fig, ax = plt.subplots() 
ax.plot(t_plot, test_x_plot, label='real') 
ax.plot(t_plot, xreverse, label='predict') 
ax.plot(x_predict, y_predict, label = 'future')
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(30))
plt.xlabel('Date')
plt.ylabel('Tourist Number')
plt.title('Predicted Tourist Number Time Series')
plt.legend()
plt.savefig('result.jpg', dpi=600)
plt.show()