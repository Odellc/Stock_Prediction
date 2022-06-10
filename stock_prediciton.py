from pickletools import optimize
from django.forms import inlineformset_factory, model_to_dict
import pandas as pd
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from keras.models import Sequential
from keras.layers import LSTM, Dropout,Dense

from sklearn.preprocessing import MinMaxScaler

#data used
#https://www.nasdaq.com/market-activity/stocks/aapl/historical


df = pd.read_csv('6_month_apple_stock.csv')

#print(df.head())
#print(df.tail(7))

#Subset the data
df = df[['Date', 'Close/Last']]
#print(df.head())

df.rename(columns={"Close/Last" :"Close"}, inplace=True)
#print(df.head())

#print(df.dtypes)

df = df.replace({'\$':''}, regex=True)

df = df.astype({"Close": float})

df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")

print(df.dtypes)

df.index = df['Date']


#Data Viz

plt.plot(df["Close"], label='Close Price History')

#plt.show()


#Data Preperation

df = df.sort_index(ascending=True, axis=0)

data = pd.DataFrame(index=range(0,len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    data['Date'][i] = df['Date'][i]
    data['Close'][i] = df['Close'][i]

print(data.head())


#Min-Max Scaler

scaler = MinMaxScaler(feature_range=(0,1))

data.index=data.Date
data.drop("Date", axis=1, inplace=True)

final_data = data.values
train_data=final_data[0:146,:]
valid_data=final_data[146:,:]

scaler=MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(final_data)
x_train, y_train = [],[]
for i in range(60, len(train_data)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])

#LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(np.shape(x_train)[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

model_data = data[len(data) - len(valid_data)-60:].values
model_data = model_data.reshape(-1, 1)
model_data = scaler.transform(model_data)


#Train and Test Data

lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(np.array(x_train), np.array(y_train), epochs=1, batch_size=1, verbose = 2)

print(model_data.shape[0])
x_test = []
for i in range(60, model_data.shape[0]):    
    x_test.append(model_data[i-60:i,0])


x_test = np.array(x_test)


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predicted_stock_price = lstm_model.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)


#Prediction Results

train_data = data[:200]
valid_data = data[200:]
valid_data['Predictions'] = predicted_stock_price
plt.plot(train_data["Close"])
plt.plot(valid_data[['Close', 'Predictions']])

plt.show()

