#M3_predicrtion6
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset (assumes 'nasdaq_data.csv' is available with the required columns)
data = pd.read_csv('1M_nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
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

# Version 1: Regularization only using XGBRegressor (no early stopping)
def run_xgb_model_reg(data):
    """
    XGB Model with Regularization:
    Base model with added L1/L2 regularization parameters.
    """
    df = data.copy()
    cols = ['Close', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_Width', 'MFI', 'EMA_20']
    for col in cols:
        df[f'{col}_lag1'] = df[col].shift(1)
    # Create additional feature: 3-day rolling mean of Close_lag1
    df['Close_rolling_mean'] = df['Close_lag1'].rolling(window=3).mean()
    df.dropna(inplace=True)
    
    feature_cols = [col for col in df.columns if 'lag1' in col] + ['Close_rolling_mean']
    
    # Split data: training (2023-2024) and testing (2025 Q1)
    train = df.loc['2023-01-01':'2024-12-31']
    test = df.loc['2025-01-01':'2025-03-31']
    X_train, y_train = train[feature_cols], train['Close']
    X_test, y_test = test[feature_cols], test['Close']
    
    model = XGBRegressor(n_estimators=200,
                         max_depth=5,
                         learning_rate=0.05,
                         gamma=1,         # regularization: minimum loss reduction to split
                         reg_alpha=0.1,   # L1 regularization
                         reg_lambda=1.0,  # L2 regularization
                         random_state=42,
                         verbosity=0)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = get_metrics(test['Close'], pd.Series(predictions, index=test.index))
    return metrics

# Version 2: Regularization with Early Stopping using xgb.train API
def run_xgb_model_reg_early(data):
    """
    XGB Model with Regularization and Early Stopping:
    Splits the training data into training and validation sets and uses xgb.train with early_stopping_rounds.
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
    
    # Create training and validation splits (80%/20% split)
    X_train_full, y_train_full = train[feature_cols], train['Close']
    split_index = int(len(X_train_full) * 0.8)
    X_train = X_train_full.iloc[:split_index]
    y_train = y_train_full.iloc[:split_index]
    X_val = X_train_full.iloc[split_index:]
    y_val = y_train_full.iloc[split_index:]
    
    # Convert to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(test[feature_cols])
    
    params = {
         'max_depth': 5,
         'learning_rate': 0.05,
         'gamma': 1,
         'reg_alpha': 0.1,
         'reg_lambda': 1.0,
         'objective': 'reg:squarederror',
         'seed': 42
    }
    evals = [(dval, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=300, evals=evals,
                    early_stopping_rounds=10, verbose_eval=False)
    predictions = bst.predict(dtest)
    metrics = get_metrics(test['Close'], pd.Series(predictions, index=test.index))
    return metrics

# Version 3: Using DART Booster with Early Stopping via xgb.train API
def run_xgb_model_dart(data):
    """
    XGB Model with DART Booster:
    Uses the 'dart' booster (which applies dropout for trees) together with regularization and early stopping.
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
    split_index = int(len(X_train_full) * 0.8)
    X_train = X_train_full.iloc[:split_index]
    y_train = y_train_full.iloc[:split_index]
    X_val = X_train_full.iloc[split_index:]
    y_val = y_train_full.iloc[split_index:]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(test[feature_cols])
    
    params = {
         'booster': 'dart',
         'max_depth': 5,
         'learning_rate': 0.1,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'gamma': 1,
         'reg_alpha': 0.1,
         'reg_lambda': 1.0,
         'objective': 'reg:squarederror',
         'seed': 42
    }
    evals = [(dval, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=300, evals=evals,
                    early_stopping_rounds=10, verbose_eval=False)
    predictions = bst.predict(dtest)
    metrics = get_metrics(test['Close'], pd.Series(predictions, index=test.index))
    return metrics

# Run all three models and compile their metrics into a comparison table
results_reg = run_xgb_model_reg(data)
results_reg_early = run_xgb_model_reg_early(data)
results_dart = run_xgb_model_dart(data)

metrics_list = ["MSE", "MAE", "R2", "DA"]

table_data = {
    "XGB_Reg": [results_reg[m] for m in metrics_list],
    "XGB_Reg_Early": [results_reg_early[m] for m in metrics_list],
    "XGB_DART": [results_dart[m] for m in metrics_list],
}

results_table = pd.DataFrame(table_data, index=metrics_list)
print("Comparison of Overfitting-Prevention Model Performance Metrics:")
print(results_table)