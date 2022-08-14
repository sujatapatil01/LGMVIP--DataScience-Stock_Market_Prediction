# LGMVIP--DataScience-Stock_Market_Prediction
# **Importing Libraries**
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

"""## **Importing Data Set**"""

url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
data = pd.read_csv(url)
data

"""## **Describing the Dataset**"""

data.describe()

data.tail()

data.dtypes

data['Date'].value_counts()

data['High'].hist()

plt.figure(figsize=(20,8))
data.plot()

data_set = data.filter(['Close'])
dataset = data.values
training_data_len=math.ceil(len(data) * 8)
training_data_len

dataset

data = data.iloc[:, 0:5]
data

training_set = data.iloc[:, 1:2].values
training_set

"""## **Scalling of Data Set**"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
data_training_scaled = scaler.fit_transform(training_set)

features_set = []
labels = []
for i in range(60, 586):
  features_set.append(data_training_scaled[i - 60:i, 0])
  labels.append(data_training_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels)

"""## **Building The LSTM**"""

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM

model = Sequential()

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(features_set, labels, epochs=50, batch_size=20)

data_testing_complete = pd.read_csv(url)
data_testing_processed = data_testing_complete.iloc[:, 1:2]
data_testing_processed

"""## **Prediction of the Data**"""

data_total = pd.concat((data['Open'], data['Open']), axis=0)

test_inputs = data_total[len(data_total) - len(data) - 60:].values
test_inputs.shape

test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)

test_feature = []
for i in range(60, 89):
  test_feature.append(test_inputs[i-60:i, 0])

test_feature = np.array(test_feature)
test_feature = np.reshape(test_feature, (test_feature.shape[0] - test_feature.shape[1], 1))
test_feature.shape

predictions = model.predict(test_feature)

predictions

x_train = data[0:1256]
y_train = data[1:1257]
print(x_train.shape)
print(y_train.shape)

x_train

np.random.seed(1)
np.random.randn(3, 3)

"""## **Drawing a Single number from the Normal Distribution**"""

np.random.normal(1)

"""## **Drawing 5 numbers from Normal Distribution**"""

np.random.normal(5)

np.random.seed(42)

np.random.normal(size=1000, scale=100).std()

"""## **Ploting Results**"""

plt.figure(figsize=(18,6))
plt.title("Stock Market Price Prediction")
plt.plot(data_testing_complete['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Total Trade Quantity', fontsize=18)
plt.show()

"""## **Analyze the Closing price from the dataframe**"""

data["Date"] = pd.to_datetime(data.Date)
data.index = data['Date']

plt.figure(figsize=(20, 10))
plt.plot(data["Open"], label='ClosePriceHist')

plt.figure(figsize=(12,6))
plt.plot(data['Date'])
plt.xlabel('Turnover (Lacs)', fontsize=18)
plt.ylabel('Total Trade Quantity', fontsize=18)
plt.show()

"""## **Analyze the Closing price from the dataframe**"""

data["Turnover (Lacs)"] = pd.to_datetime(data.Date)
data.index = data['Turnover (Lacs)']

plt.figure(figsize=(20, 10))
plt.plot(data["Turnover (Lacs)"], label='ClosePriceHist')

sns.set(rc = {'figure.figsize': (20, 5)})
data['Open'].plot(linewidth = 1,color='blue')

data.columns

df = pd.read_csv(url)
df

cols_plot = ['Open','High','Low','Last','Close']
axes = df[cols_plot].plot(alpha = 1, figsize=(20, 30), subplots = True)

for ax in axes:
    ax.set_ylabel('Variation')
