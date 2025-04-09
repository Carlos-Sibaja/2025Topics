# M3_prediction5
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset (assumes 'nasdaq_data.csv' is available with the required columns)
data = pd.read_csv('nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
data.index = pd.to_datetime(data.index, utc=True)
data.index = data.index.tz_convert(None)

# Remove old predictions if exist
data = data.drop(columns=['Close_Predicted_RF', 'Close_Predicted_XGB'], errors='ignore')
print("===== Dataset Loaded =====")

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

# Model 1: Baseline XGB with lag features and a 3-day rolling mean
def run_xgb_model_1(data):
    """
    XGB Model Version 1: Baseline parameters with lag features and a 3-day rolling mean.
    """
    df = data.copy()
    cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 'MFI', 'EMA_20']
    for col in cols:
        df[f'{col}_lag1'] = df[col].shift(1)
    # Additional feature: 3-day rolling mean of Close_lag1
    df['Close_rolling_mean'] = df['Close_lag1'].rolling(window=3).mean()
    df.dropna(inplace=True)
    
    feature_cols = [col for col in df.columns if 'lag1' in col] + ['Close_rolling_mean']
    
    # Split data into training (2023-2024) and testing (2025 Q1)
    train = df.loc['2023-01-01':'2024-12-31']
    test = df.loc['2025-01-01':'2025-03-31']
    X_train, y_train = train[feature_cols], train['Close']
    X_test, y_test = test[feature_cols], test['Close']
    
    # Baseline XGB parameters
    model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, 
                         random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = get_metrics(y_test, pd.Series(predictions, index=y_test.index))
    return metrics

# Model 2: Tuned hyperparameters with lag features and a 3-day rolling mean
def run_xgb_model_2(data):
    """
    XGB Model Version 2: Tuned hyperparameters with lag features and a 3-day rolling mean.
    """
    df = data.copy()
    cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 'MFI', 'EMA_20']
    for col in cols:
        df[f'{col}_lag1'] = df[col].shift(1)
    df['Close_rolling_mean'] = df['Close_lag1'].rolling(window=3).mean()
    df.dropna(inplace=True)
    
    feature_cols = [col for col in df.columns if 'lag1' in col] + ['Close_rolling_mean']
    
    train = df.loc['2023-01-01':'2024-12-31']
    test = df.loc['2025-01-01':'2025-03-31']
    X_train, y_train = train[feature_cols], train['Close']
    X_test, y_test = test[feature_cols], test['Close']
    
    # Tuned XGB parameters
    model = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = get_metrics(y_test, pd.Series(predictions, index=y_test.index))
    return metrics

# Model 3: Using early stopping with a validation split via xgb.train
def run_xgb_model_3(data):
    """
    XGB Model Version 3: Incorporates early stopping using the xgb.train API with a validation split.
    Uses lag features and a 3-day rolling mean.
    """
    df = data.copy()
    cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 'MFI', 'EMA_20']
    for col in cols:
        df[f'{col}_lag1'] = df[col].shift(1)
    df['Close_rolling_mean'] = df['Close_lag1'].rolling(window=3).mean()
    df.dropna(inplace=True)
    
    feature_cols = [col for col in df.columns if 'lag1' in col] + ['Close_rolling_mean']
    
    train = df.loc['2023-01-01':'2024-12-31']
    test = df.loc['2025-01-01':'2025-03-31']
    X_train_full, y_train_full = train[feature_cols], train['Close']
    
    # Time-series split: 80% for training, 20% for validation
    split_index = int(len(X_train_full) * 0.8)
    X_train = X_train_full.iloc[:split_index]
    y_train = y_train_full.iloc[:split_index]
    X_val = X_train_full.iloc[split_index:]
    y_val = y_train_full.iloc[split_index:]
    
    # Convert training and validation sets to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(test[feature_cols])
    
    params = {
        'max_depth': 7,
        'learning_rate': 0.05,
        'objective': 'reg:squarederror',
        'seed': 42
    }
    evals = [(dval, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=500, evals=evals,
                    early_stopping_rounds=10, verbose_eval=False)
    predictions = bst.predict(dtest)
    metrics = get_metrics(test['Close'], pd.Series(predictions, index=test.index))
    return metrics

# Model 4: Normalized features with tuned hyperparameters
def run_xgb_model_4(data):
    """
    XGB Model Version 4: Applies StandardScaler normalization to features and uses tuned hyperparameters.
    Uses lag features and a 3-day rolling mean.
    """
    df = data.copy()
    cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 'MFI', 'EMA_20']
    for col in cols:
        df[f'{col}_lag1'] = df[col].shift(1)
    df['Close_rolling_mean'] = df['Close_lag1'].rolling(window=3).mean()
    df.dropna(inplace=True)
    
    feature_cols = [col for col in df.columns if 'lag1' in col] + ['Close_rolling_mean']
    
    train = df.loc['2023-01-01':'2024-12-31']
    test = df.loc['2025-01-01':'2025-03-31']
    X_train, y_train = train[feature_cols], train['Close']
    X_test, y_test = test[feature_cols], test['Close']
    
    # Apply feature normalization
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                  index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                                 index=X_test.index, columns=X_test.columns)
    
    model = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, verbosity=0)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    metrics = get_metrics(y_test, pd.Series(predictions, index=y_test.index))
    return metrics

# Run all four XGB models and compile their metrics into a comparison table
results_xgb1 = run_xgb_model_1(data)
results_xgb2 = run_xgb_model_2(data)
results_xgb3 = run_xgb_model_3(data)
results_xgb4 = run_xgb_model_4(data)

# Define the metric names for rows
metrics_list = ["MSE", "MAE", "R2", "DA"]

# Create a DataFrame to compare the performance of the four models
table_data = {
    "XGB_Model1": [results_xgb1[m] for m in metrics_list],
    "XGB_Model2": [results_xgb2[m] for m in metrics_list],
    "XGB_Model3": [results_xgb3[m] for m in metrics_list],
    "XGB_Model4": [results_xgb4[m] for m in metrics_list],
}

results_table = pd.DataFrame(table_data, index=metrics_list)
print("Comparison of XGB Model Performance Metrics:")
print(results_table)


# ===============================
# Additional: Train vs Test Evaluation (Base Model)
# ===============================

# Re-run Base Model to get training predictions
df = data.copy()
cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 'MFI', 'EMA_20']
for col in cols:
    df[f'{col}_lag1'] = df[col].shift(1)
df['Close_rolling_mean'] = df['Close_lag1'].rolling(window=3).mean()
df.dropna(inplace=True)

feature_cols = [col for col in df.columns if 'lag1' in col] + ['Close_rolling_mean']

train = df.loc['2023-01-01':'2024-12-31']
test = df.loc['2025-01-01':'2025-03-31']
X_train, y_train = train[feature_cols], train['Close']
X_test, y_test = test[feature_cols], test['Close']

model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, 
                     random_state=42, verbosity=0)
model.fit(X_train, y_train)

# Get train and test predictions
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# Compute metrics
train_metrics = {
    "MSE": mean_squared_error(y_train, pred_train),
    "MAE": mean_absolute_error(y_train, pred_train),
    "R2": r2_score(y_train, pred_train)
}

test_metrics = {
    "MSE": mean_squared_error(y_test, pred_test),
    "MAE": mean_absolute_error(y_test, pred_test),
    "R2": r2_score(y_test, pred_test)
}

# Create a comparison table
comparison_table = pd.DataFrame({
    "Training": train_metrics,
    "Testing": test_metrics,
    "Difference": {k: train_metrics[k] - test_metrics[k] for k in train_metrics}
})

print("\n=== Train vs Test Performance (Base Model) ===")
print(comparison_table)
