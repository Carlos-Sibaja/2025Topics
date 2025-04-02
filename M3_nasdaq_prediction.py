#M3 -REGRESSION MODEL
# ===============================
# Import necessary libraries
# ===============================
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ===============================
# Load dataset with precomputed indicators
# ===============================
df = pd.read_csv('nasdaq_data.csv', parse_dates=['Date'], index_col='Date')

print("===== Dataset loaded successfully =====")
print(df.head())

# ===============================
# Prepare features and target
# ===============================

# Feature columns
features = ['Open', 'High', 'Low', 'Close', 'Volume', 
            'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 
            'MFI', 'EMA_20']

# Define target
target = 'Close'  # Predicting next day's Close

# Shift target by -1 to predict next day close price
df['Target'] = df['Close'].shift(-1)

# Remove last row due to NaN after shifting
df.dropna(inplace=True)

# Define X and y
X = df[features]
y = df['Target']

# ===============================
# Time Series Split
# ===============================
tscv = TimeSeriesSplit(n_splits=5)

# ===============================
# Random Forest + GridSearchCV
# ===============================
params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [5, 10]
}

rf = RandomForestRegressor(random_state=42)
gsearch = GridSearchCV(rf, params, cv=tscv, scoring='neg_mean_squared_error')
gsearch.fit(X, y)

print("\nBest parameters found:", gsearch.best_params_)

# ===============================
# Model Evaluation
# ===============================

# Predictions
y_pred = gsearch.predict(X)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.4f}")

# ===============================
# Feature Importance
# ===============================
importances = pd.Series(gsearch.best_estimator_.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', figsize=(8,5))
plt.title("Feature Importances")
plt.show()

# ===============================
# Prediction Plot
# ===============================
plt.figure(figsize=(15,5))
plt.plot(df.index, y, label='Actual Close')
plt.plot(df.index, y_pred, label='Predicted Close', alpha=0.7)
plt.legend()
plt.title("Random Forest - Actual vs Predicted Close Price")
plt.show()
