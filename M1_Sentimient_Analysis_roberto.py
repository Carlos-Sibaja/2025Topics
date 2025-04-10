# ===============================
# Import Libraries
# ===============================
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ===============================
# Load the Enhanced NASDAQ Dataset
# ===============================
df = pd.read_csv('nasdaq_roberto.csv', parse_dates=['Date'], index_col='Date')

# ===============================
# Create Target Variable: Next Day's Closing Price
# ===============================
df['Target_Close'] = df['Close'].shift(-1)
df.dropna(inplace=True)  # Drop last row with NaN target

# ===============================
# Feature Selection
# ===============================
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'Sentiment_T1', 'Sentiment_T2', 'Sentiment_T3', 'Sentiment_3DayAVG']

X = df[features]
y = df['Target_Close']

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ===============================
# Train XGBoost Regressor
# ===============================
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ===============================
# Predict and Evaluate
# ===============================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Mean Squared Error: {mse:.4f}")
print(f"🧮 R^2 Score: {r2:.4f}")

# ===============================
# Plot Actual vs Predicted
# ===============================
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted', linewidth=2)
plt.title('NASDAQ Close Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()