#M3_prediction3
# Crea retornos como variable target (más estable que precios).

# Añade lags de 1 día a indicadores y precios.

# Separa 80% train y 20% test de forma estricta (simulando "futuro" nunca visto).

# Evalúa con:

# MSE

# MAE

# R2

# Directional Accuracy (clave en finanzas)

# Compara Random Forest y XGBoost.

# Genera gráficas de importancia de variables y predicciones.


# ===============================
# Import Libraries
# ===============================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv('1M_nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
df.index = pd.to_datetime(df.index, utc=True)
df.index = df.index.tz_convert(None)

# Remove old predictions if exist
df = df.drop(columns=['Close_Predicted_RF', 'Close_Predicted_XGB'], errors='ignore')

print("===== Dataset Loaded =====")

# ===============================
# Feature Engineering
# ===============================
for col in ['Close', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 'MFI', 'EMA_20']:
    df[f'{col}_lag1'] = df[col].shift(1)

df.dropna(inplace=True)

# ===============================
# Train-Test Split (2023-2024 only for training)
# ===============================
train = df.loc['2023-01-01':'2024-12-31'].copy()
future = df.loc['2025-01-01':'2025-03-31'].copy()

features = [col for col in df.columns if 'lag1' in col]
X_train = train[features]
y_train = train['Close']

X_future = future[features]

# ===============================
# Random Forest Model
# ===============================
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
rf.fit(X_train, y_train)
future['Close_Predicted_RF'] = rf.predict(X_future)

# ===============================
# XGBoost Model
# ===============================
xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
xgb.fit(X_train, y_train)
future['Close_Predicted_XGB'] = xgb.predict(X_future)

# ===============================
# Append Predictions to CSV
# ===============================
df = df.merge(future[['Close_Predicted_RF', 'Close_Predicted_XGB']], left_index=True, right_index=True, how='outer')
df.to_csv('nasdaq_data.csv')
print("\nPredictions added and nasdaq_data.csv updated.")

# ===============================
# Plot Predictions
# ===============================
plt.figure(figsize=(15,6))
plt.plot(df.loc['2025-01-01':'2025-03-31'].index, df.loc['2025-01-01':'2025-03-31']['Close'], label='Close Real', linewidth=2)
plt.plot(df.loc['2025-01-01':'2025-03-31'].index, df.loc['2025-01-01':'2025-03-31']['Close_Predicted_RF'], label='Predicted RF', linestyle='--')
plt.plot(df.loc['2025-01-01':'2025-03-31'].index, df.loc['2025-01-01':'2025-03-31']['Close_Predicted_XGB'], label='Predicted XGBoost', linestyle='--')
plt.legend()
plt.title("NASDAQ Close Price Prediction (Jan-Mar 2025)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# ===============================
# Display final prediction table (Jan-Mar 2025)
# ===============================

# Create a dataframe for display
result = df.loc['2025-01-01':'2025-03-31', ['Close', 'Close_Predicted_RF', 'Close_Predicted_XGB']].copy()
result.reset_index(inplace=True)
result.rename(columns={'Date': 'Date', 'Close': 'Close Real'}, inplace=True)

# Show full table in terminal without cutting off
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("\n===== Predicciones de Cierre (Enero-Marzo 2025) =====")
    print(result)


    # ===============================
# Evaluation of Prediction Quality
# ===============================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_predictions(true, pred, name):
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    da = np.mean((np.sign(true.diff()) == np.sign(pred.diff()))) * 100

    print(f"\n===== Quality Indicator for: {name} =====")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Accuracy Direccional (DA): {da:.2f}%")

# ===============================
# Evaluate both models
# ===============================

print("\n====== Model Evaluation ======")
future = future.dropna(subset=['Close'])  # aseguramos que Close no sea NaN

evaluate_predictions(future['Close'], future['Close_Predicted_RF'], "Random Forest")
evaluate_predictions(future['Close'], future['Close_Predicted_XGB'], "XGBoost")