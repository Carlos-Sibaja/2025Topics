import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a ticker object for the stock
ticker = yf.Ticker("AAPL")

# Fetch historical stock data from January 1, 2020 to January 1, 2023
historical_data = ticker.history(start='2020-01-01', end='2023-01-01')

# Display the retrieved data
print(historical_data)

## Calculatin a 20-day moving average closing price
moving_average = historical_data['Close'].rolling(window=20).mean()

# Add the moving average as a new column in the DataFrame
historical_data['20 Day MA'] = moving_average

# Display the updated DataFrame
print(historical_data.tail())

# Plot the closing prices and the 20-day moving average
plt.figure(figsize=(12, 6))
plt.plot(historical_data.index, historical_data['Close'], label='Close Price')
plt.plot(historical_data.index, historical_data['20 Day MA'], label='20 Day MA', linestyle='--')
plt.title('Apple Inc. Stock Price & 20-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()