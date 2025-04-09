# preditions4

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

def get_metrics(true, pred):
    """Calculate evaluation metrics for the predictions."""
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    # Compute directional accuracy: percentage of days where the sign of the difference matches
    true_diff = true.diff().dropna()
    pred_diff = pred.diff().dropna()
    da = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100
    return {"MSE": mse, "MAE": mae, "R2": r2, "DA": da}

def run_version1(data):
    """
    Version 1: Feature Aggregation.
    Adds an extra feature: the 3-day rolling mean of the previous day's Close (Close_lag1).
    """
    df = data.copy()
    cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 'MFI', 'EMA_20']
    for col in cols:
        df[f'{col}_lag1'] = df[col].shift(1)
    df.dropna(inplace=True)
    
    # Additional feature: 3-day rolling mean of Close_lag1
    df['Close_rolling_mean'] = df['Close_lag1'].rolling(window=3).mean()
    df.dropna(inplace=True)
    
    # Define feature set (all lag features + new rolling mean)
    feature_cols = [col for col in df.columns if 'lag1' in col] + ['Close_rolling_mean']
    
    # Split data into training (2023-2024) and future test (2025 Q1)
    train = df.loc['2023-01-01':'2024-12-31']
    test = df.loc['2025-01-01':'2025-03-31']
    X_train, y_train = train[feature_cols], train['Close']
    X_test, y_test = test[feature_cols], test['Close']
    
    # Train models with baseline parameters
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    
    xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    
    metrics_rf = get_metrics(y_test, pd.Series(pred_rf, index=y_test.index))
    metrics_xgb = get_metrics(y_test, pd.Series(pred_xgb, index=y_test.index))
    
    return {"RF": metrics_rf, "XGB": metrics_xgb}

def run_version2(data):
    """
    Version 2: Normalization.
    Scales the lag features using StandardScaler.
    """
    df = data.copy()
    cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 'MFI', 'EMA_20']
    for col in cols:
        df[f'{col}_lag1'] = df[col].shift(1)
    df.dropna(inplace=True)
    
    feature_cols = [col for col in df.columns if 'lag1' in col]
    
    # Split data into training (2023-2024) and test (2025 Q1)
    train = df.loc['2023-01-01':'2024-12-31']
    test = df.loc['2025-01-01':'2025-03-31']
    X_train, y_train = train[feature_cols], train['Close']
    X_test, y_test = test[feature_cols], test['Close']
    
    # Apply StandardScaler to the features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    # Train models with baseline parameters on normalized data
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
    rf.fit(X_train_scaled, y_train)
    pred_rf = rf.predict(X_test_scaled)
    
    xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
    xgb.fit(X_train_scaled, y_train)
    pred_xgb = xgb.predict(X_test_scaled)
    
    metrics_rf = get_metrics(y_test, pd.Series(pred_rf, index=y_test.index))
    metrics_xgb = get_metrics(y_test, pd.Series(pred_xgb, index=y_test.index))
    
    return {"RF": metrics_rf, "XGB": metrics_xgb}

def run_version3(data):
    """
    Version 3: Hyperparameter Tuning.
    Uses adjusted hyperparameters for both Random Forest and XGBoost.
    """
    df = data.copy()
    cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 'MFI', 'EMA_20']
    for col in cols:
        df[f'{col}_lag1'] = df[col].shift(1)
    df.dropna(inplace=True)
    
    feature_cols = [col for col in df.columns if 'lag1' in col]
    
    # Split data into training (2023-2024) and test (2025 Q1)
    train = df.loc['2023-01-01':'2024-12-31']
    test = df.loc['2025-01-01':'2025-03-31']
    X_train, y_train = train[feature_cols], train['Close']
    X_test, y_test = test[feature_cols], test['Close']
    
    # Train models with tuned hyperparameters
    rf = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_leaf=3, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    
    xgb = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.1, subsample=0.8,
                       colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)
    
    metrics_rf = get_metrics(y_test, pd.Series(pred_rf, index=y_test.index))
    metrics_xgb = get_metrics(y_test, pd.Series(pred_xgb, index=y_test.index))
    
    return {"RF": metrics_rf, "XGB": metrics_xgb}

# Main execution
if __name__ == "__main__":
    # Load the dataset (assumes 'nasdaq_data.csv' is available with the required columns)
    data = pd.read_csv('nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
    data.index = pd.to_datetime(data.index, utc=True)
    data.index = data.index.tz_convert(None)

# Remove old predictions if exist
    data = data.drop(columns=['Close_Predicted_RF', 'Close_Predicted_XGB'], errors='ignore')

    print("===== Dataset Loaded =====")
    
    
    # Run all three versions
    results_v1 = run_version1(data)
    results_v2 = run_version2(data)
    results_v3 = run_version3(data)
    
    # Prepare a results table with rows as indicators and columns as each run/model
    metrics_list = ["MSE", "MAE", "R2", "DA"]
    table_data = {
        "Run1_Random": [results_v1["RF"][m] for m in metrics_list],
        "Run1_XGB": [results_v1["XGB"][m] for m in metrics_list],
        "Run2_Random": [results_v2["RF"][m] for m in metrics_list],
        "Run2_XGB": [results_v2["XGB"][m] for m in metrics_list],
        "Run3_Random": [results_v3["RF"][m] for m in metrics_list],
        "Run3_XGB": [results_v3["XGB"][m] for m in metrics_list],
    }
    results_table = pd.DataFrame(table_data, index=metrics_list)
    
    # Display the results table
    print("Comparison of Model Performance Metrics:")
    print(results_table)
