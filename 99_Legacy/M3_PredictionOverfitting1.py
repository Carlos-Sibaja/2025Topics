#M3_PredictionOverfitting1
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ===== Load Dataset =====
data = pd.read_csv('1M_nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
data.index = pd.to_datetime(data.index, utc=True)
data.index = data.index.tz_convert(None)
data = data.drop(columns=['Close_Predicted_RF', 'Close_Predicted_XGB'], errors='ignore')
print("===== Dataset Loaded =====")

def get_metrics(true, pred):
    pred = pd.Series(pred, index=true.index) 
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    true_diff = true.diff().dropna()
    pred_diff = pred.diff().dropna()
    da = np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100
    return {"MSE": mse, "MAE": mae, "R2": r2, "DA": da}

def prepare_data(data):
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
    return X_train, y_train, X_test, y_test

# ========= MODEL 1: Base =========
print("\n=== MODEL 1: BASE ===")
X_train, y_train, X_test, y_test = prepare_data(data)
model1 = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
model1.fit(X_train, y_train)
pred_train = model1.predict(X_train)
pred_test = model1.predict(X_test)
train_metrics = get_metrics(y_train, pred_train)
test_metrics = get_metrics(y_test, pred_test)
table1 = pd.DataFrame({"Training": train_metrics, "Testing": test_metrics, "Difference": {k: train_metrics[k] - test_metrics[k] for k in train_metrics}})
print(table1)

# ========= MODEL 2: Base + Regularization =========
print("\n=== MODEL 2: BASE + REGULARIZATION ===")
model2 = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                      gamma=1, reg_alpha=0.5, reg_lambda=1,
                      random_state=42, verbosity=0)
model2.fit(X_train, y_train)
pred_train = model2.predict(X_train)
pred_test = model2.predict(X_test)
train_metrics = get_metrics(y_train, pred_train)
test_metrics = get_metrics(y_test, pred_test)
table2 = pd.DataFrame({"Training": train_metrics, "Testing": test_metrics, "Difference": {k: train_metrics[k] - test_metrics[k] for k in train_metrics}})
print(table2)

# ========= MODEL 3: Base + Regularization + Early Stopping =========
print("\n=== MODEL 3: BASE + REGULARIZATION + EARLY STOPPING ===")
split_index = int(len(X_train) * 0.8)
X_train_part, X_val = X_train.iloc[:split_index], X_train.iloc[split_index:]
y_train_part, y_val = y_train.iloc[:split_index], y_train.iloc[split_index:]
dtrain = xgb.DMatrix(X_train_part, label=y_train_part)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

params = {
    'max_depth': 5, 'learning_rate': 0.05,
    'gamma': 1, 'reg_alpha': 0.5, 'reg_lambda': 1,
    'objective': 'reg:squarederror', 'seed': 42
}
evals = [(dval, 'eval')]
model3 = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=10, verbose_eval=False)
pred_train = model3.predict(xgb.DMatrix(X_train))
pred_test = model3.predict(dtest)
train_metrics = get_metrics(y_train, pred_train)
test_metrics = get_metrics(y_test, pred_test)
table3 = pd.DataFrame({"Training": train_metrics, "Testing": test_metrics, "Difference": {k: train_metrics[k] - test_metrics[k] for k in train_metrics}})
print(table3)

# ========= MODEL 4: Base + Regularization + Early Stopping + Normalization =========
print("\n=== MODEL 4: BASE + REGULARIZATION + EARLY STOPPING + NORMALIZATION ===")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

X_train_part, X_val = X_train_scaled.iloc[:split_index], X_train_scaled.iloc[split_index:]
y_train_part, y_val = y_train.iloc[:split_index], y_train.iloc[split_index:]

dtrain = xgb.DMatrix(X_train_part, label=y_train_part)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test_scaled)

model4 = xgb.train(params, dtrain, num_boost_round=500, evals=evals, early_stopping_rounds=10, verbose_eval=False)
pred_train = model4.predict(xgb.DMatrix(X_train_scaled))
pred_test = model4.predict(dtest)
train_metrics = get_metrics(y_train, pred_train)
test_metrics = get_metrics(y_test, pred_test)
table4 = pd.DataFrame({"Training": train_metrics, "Testing": test_metrics, "Difference": {k: train_metrics[k] - test_metrics[k] for k in train_metrics}})
print(table4)