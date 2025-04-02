#nasdaq_extractor
# ===============================
# Import libraries
# ===============================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

# ===============================
# Download historical data for NASDAQ
# ===============================

# Create a ticker object for NASDAQ Composite Index
ticker = yf.Ticker("^IXIC")

# Fetch historical data from January 1, 2021 to December 31, 2024
nasdaq_data = ticker.history(start='2021-01-01', end='2024-12-31')

# Display the raw data
print("===== Raw NASDAQ Data =====")
print(nasdaq_data.head())

# ===============================
# Calculate technical indicators
# ===============================

# RSI (Relative Strength Index)
nasdaq_data['RSI'] = ta.momentum.RSIIndicator(close=nasdaq_data['Close'], window=14).rsi()

# MACD and MACD Signal
macd = ta.trend.MACD(close=nasdaq_data['Close'])
nasdaq_data['MACD'] = macd.macd()
nasdaq_data['MACD_Signal'] = macd.macd_signal()

# Bollinger Bands Width
bollinger = ta.volatility.BollingerBands(close=nasdaq_data['Close'], window=20)
nasdaq_data['Bollinger_Width'] = bollinger.bollinger_hband() - bollinger.bollinger_lband()

# Money Flow Index (MFI)
nasdaq_data['MFI'] = ta.volume.MFIIndicator(high=nasdaq_data['High'], 
                                            low=nasdaq_data['Low'], 
                                            close=nasdaq_data['Close'], 
                                            volume=nasdaq_data['Volume'], 
                                            window=14).money_flow_index()

# Exponential Moving Average (EMA) 20 days
nasdaq_data['EMA_20'] = ta.trend.EMAIndicator(close=nasdaq_data['Close'], window=20).ema_indicator()

# ===============================
# Remove missing values
# ===============================
nasdaq_data.dropna(inplace=True)

# ===============================
# Display dataframe with indicators
# ===============================
print("\n===== NASDAQ Data with Technical Indicators =====")
print(nasdaq_data.tail())

# ===============================
# Save dataframe to CSV
# ===============================
nasdaq_data.to_csv('nasdaq_data.csv')
print("\n'nasdaq_data.csv' saved successfully!")

# ===============================
# Plot example: Close Price and EMA
# ===============================

plt.figure(figsize=(12, 6))
plt.plot(nasdaq_data.index, nasdaq_data['Close'], label='Close Price')
plt.plot(nasdaq_data.index, nasdaq_data['EMA_20'], label='EMA 20', linestyle='--')
plt.title('NASDAQ Composite - Close Price & EMA(20)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
