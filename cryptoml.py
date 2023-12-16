import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cryptocompare
import datetime as dt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
symbol = 'BTC'
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

# Fetch data with improved error handling
try:
    data = cryptocompare.get_historical_price_day(symbol, currency='USD', toTs=end, limit=2000)
    
    # Convert the list of dictionaries to a DataFrame
    data = pd.DataFrame(data)
    
    # Print the raw data fetched
    print("Raw data fetched:")
    print(data)

    # Convert 'close' values to integers
    data['close'] = data['close'].astype(int)
    
    print("Fetched data successfully:")
    print(data.head())
except (ValueError, KeyError, TypeError) as e:
    print(f"Error processing data: {e}")
    data = None  # Set data to None to indicate the error

# Further diagnostics
print(f"Type of 'data': {type(data)}")
if isinstance(data, pd.DataFrame):
    print(f"Columns of 'data': {data.columns}")
    print(f"Shape of 'data': {data.shape}")
    print(f"Data types of 'data': {data.dtypes}")

# Continue only if data is not None and it's a DataFrame
if isinstance(data, pd.DataFrame):
    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    try:
        scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
        print("Data successfully scaled.")
    except Exception as e:
        print(f"Error scaling data: {e}")
    
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
    
    # For CryptoCompare:
    test_data = cryptocompare.get_historical_price_day(symbol, currency='USD', toTs=test_end, limit=200)
    test_data = pd.DataFrame(test_data)  # Convert test data to DataFrame
    
    actual_prices = test_data['close'].values

    total_dataset = pd.concat((data['close'], test_data['close']), axis=0)

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
    plt.plot(actual_prices, color='black', label=f"Actual {symbol} Price")
    plt.plot(predicted_price, color='green', label=f"Predicted {symbol} Price")
    plt.title(f"{symbol} Price Prediction")
    plt.xlabel('Time')
    plt.ylabel(f'{symbol} Price')
    plt.legend()
    plt.show()
else:
    print("Data is not in the expected format. Check the fetched data.")
