#M3_PredictionOverfitting5
# M3_Overfitting_v8.py — Walk-Forward Version

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ===== Load Dataset =====
data = pd.read_csv('nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
data.index = pd.to_datetime(data.index, utc=True)
data.index = data.index.tz_convert(None)
data = data.drop(columns=['Close_Predicted_RF', 'Close_Predicted_XGB'], errors='ignore')
print("===== Dataset Loaded =====")

# ===== Feature Engineering =====
df = data.copy()
df['Return'] = df['Close'].pct_change()
df['Volatility'] = df['Return'].rolling(5).std()
for col in ['Close', 'RSI', 'MACD', 'MFI']:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)

df['Rolling_Return'] = df['Return'].rolling(3).sum().shift(-3)
df['Target'] = np.where(df['Rolling_Return'] > 0.003, 1,
                        np.where(df['Rolling_Return'] < -0.003, 0, np.nan))

df.dropna(inplace=True)
print(f"Target distribution:\n{df['Target'].value_counts(normalize=True) * 100}")

features = [col for col in df.columns if ('lag' in col) or (col in ['Return', 'Volatility'])]

# ===== Setup =====
train = df.loc['2019-01-01':'2024-12-31']
test = df.loc['2025-01-01':'2025-03-31']
X_train, y_train = train[features], train['Target']

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ===== Model (Same as Model 4) =====
model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03,
                      gamma=2, reg_alpha=1, reg_lambda=1.5,
                      scale_pos_weight=scale_pos_weight,
                      random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")

model.fit(X_train, y_train)

# ====== Walk-Forward Test ======
print("\n===== WALK-FORWARD TEST =====")

results = []
for month in ['2025-01', '2025-02', '2025-03']:
    fold = df.loc[month]
    X_fold = fold[features]
    y_fold = fold['Target']
    if len(y_fold) == 0:
        continue  # Skip empty months
    y_pred = model.predict(X_fold)
    metrics = {
        "Month": month,
        "DA": accuracy_score(y_fold, y_pred) * 100,
        "Precision": precision_score(y_fold, y_pred) * 100,
        "Recall": recall_score(y_fold, y_pred) * 100,
        "ROC_AUC": roc_auc_score(y_fold, y_pred) * 100,
        "Samples": len(y_fold)
    }
    results.append(metrics)

results_df = pd.DataFrame(results)
print(results_df)

# ====== Average Performance ======
print("\n=== AVERAGE PERFORMANCE ===")
print(results_df[["DA", "Precision", "Recall", "ROC_AUC"]].mean())

