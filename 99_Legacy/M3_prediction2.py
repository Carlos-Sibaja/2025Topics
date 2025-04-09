#M3 prediction2.
# ==================================
# NASDAQ Random Forest Regression (Improved Version)
# ==================================

# Cell 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# ==================================
# Cell 2: Load pre-processed dataset
# ==================================

df = pd.read_csv('nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
print("Dataset loaded successfully.")
print(df.head())

# ==================================
# Cell 3: Feature engineering (lagged indicators & target)
# ==================================

# Compute log-return as target
df['Log_Return'] = np.log(df['Close']).diff().shift(-1)

# Create lagged indicators (t-1)
for col in ['Close', 'Open', 'High', 'Low', 'Volume', 
            'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 
            'MFI', 'EMA_20']:
    df[f'{col}_lag1'] = df[col].shift(1)

# Drop missing values
df.dropna(inplace=True)

# ==================================
# Cell 4: Prepare features and target
# ==================================

# Use only lagged variables to predict future return
lagged_features = [f'{col}_lag1' for col in ['Close', 'Open', 'High', 'Low', 'Volume',
                                             'RSI', 'MACD', 'MACD_Signal', 
                                             'Bollinger_Width', 'MFI', 'EMA_20']]

X = df[lagged_features]
y = df['Log_Return']

print(f"Dataset ready for modeling. X shape: {X.shape}, y shape: {y.shape}")

# ==================================
# Cell 5: TimeSeriesSplit + GridSearch
# ==================================

tscv = TimeSeriesSplit(n_splits=5)

params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_leaf': [5, 10]
}

rf = RandomForestRegressor(random_state=42)
gsearch = GridSearchCV(rf, params, cv=tscv, scoring='neg_mean_squared_error')
gsearch.fit(X, y)

print("Best parameters found:")
print(gsearch.best_params_)

# ==================================
# Cell 6: Evaluation
# ==================================

y_pred = gsearch.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.6f}")
print(f"R-squared (R2): {r2:.4f}")

# ==================================
# Cell 7: Feature Importance
# ==================================

importances = pd.Series(gsearch.best_estimator_.feature_importances_, index=lagged_features)
importances.sort_values().plot(kind='barh', figsize=(8,5))
plt.title("Feature Importances (Predicting Log-Returns)")
plt.show()

# ==================================
# Cell 8: Plot predicted vs actual returns
# ==================================

plt.figure(figsize=(15,5))
plt.plot(df.index, y, label='Actual Log-Return')
plt.plot(df.index, y_pred, label='Predicted Log-Return', alpha=0.7)
plt.legend()
plt.title("Random Forest: Actual vs Predicted Log-Returns")
plt.show()
