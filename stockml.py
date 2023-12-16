
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
company = 'AAPL'

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

# Fetch data from Yahoo Finance with improved error handling
try:
    data = yf.download(company, start=start, end=end)
    
    # Convert 'Close' values to integers
    data['Close'] = data['Close'].astype(int)
    
    print("Fetched data successfully:")
    print(data.head())
except (ValueError, KeyError, TypeError) as e:
    print(f"Error processing data: {e}")
    data = None  # Set data to None to indicate the error

# Further diagnostics
print(f"Type of 'data': {type(data)}")
if isinstance(data, pd.DataFrame):
    print(f"Columns of 'data': {data.columns}")

# Continue only if data is not None and it's a DataFrame
if isinstance(data, pd.DataFrame):
    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of the next closing value

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Test the data
    test_start = dt.datetime(2020, 1, 1)
    test_end = dt.datetime.now()
    test_data = yf.download(company, start=test_start, end=test_end)
    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)

    # Scale and round model inputs to integers
    model_inputs = np.round(scaler.transform(model_inputs)).astype(int)

    # Prediction on test data
    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_price = model.predict(x_test)
    predicted_price = np.round(scaler.inverse_transform(predicted_price)).astype(int)

    # Plot
    plt.plot(actual_prices, color='black', label=f"Actual {company} Price")
    plt.plot(predicted_price, color='green', label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()
else:
    print("Data is not in the expected format. Check the fetched data.")