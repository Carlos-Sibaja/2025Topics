#M3_PredictionOverfitting3
# M3 Overfitting v5 - BASE = Model 2 + new variations

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
df['Return_volatility'] = df['Return'].rolling(5).std()
df['EMA_10'] = df['Close'].ewm(span=10).mean()
df['EMA_50'] = df['Close'].ewm(span=50).mean()

for col in ['Close', 'EMA_10', 'EMA_50', 'RSI', 'MACD', 'MFI']:
    df[f'{col}_lag1'] = df[col].shift(1)

# ===== Target: Filter significant movements only =====
df['Rolling_Return'] = df['Return'].rolling(3).sum().shift(-3)
df['Target'] = np.where(df['Rolling_Return'] > 0.003, 1,
                        np.where(df['Rolling_Return'] < -0.003, 0, np.nan))
df.dropna(inplace=True)
print(f"Target distribution:\n{df['Target'].value_counts(normalize=True) * 100}")

features = [col for col in df.columns if 'lag1' in col] + ['Return', 'Return_volatility']

# ===== Split Train / Test =====
train = df.loc['2023-01-01':'2024-12-31']
test = df.loc['2025-01-01':'2025-03-31']
X_train, y_train = train[features], train['Target']
X_test, y_test = test[features], test['Target']

# ===== Metrics =====
def get_classification_metrics(y_true, y_pred):
    return {
        "Accuracy (DA)": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred) * 100,
        "Recall": recall_score(y_true, y_pred) * 100,
        "ROC_AUC": roc_auc_score(y_true, y_pred) * 100
    }

# ================================
# === MODEL 1: BASE (Former Model 2) ===
# ================================

print("\n=== MODEL 1: BASE (Previous Model 2) ===")
model1 = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                       gamma=1, reg_alpha=0.5, reg_lambda=1,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
model1.fit(X_train, y_train)
pred_train = model1.predict(X_train)
pred_test = model1.predict(X_test)
print(pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                    "Testing": get_classification_metrics(y_test, pred_test)}))

# ================================
# === MODEL 2: BASE + Class Balancing ===
# ================================

print("\n=== MODEL 2: BASE + Balanced Classes ===")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model2 = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                       gamma=1, reg_alpha=0.5, reg_lambda=1,
                       scale_pos_weight=scale_pos_weight,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")

model2.fit(X_train, y_train)
pred_train = model2.predict(X_train)
pred_test = model2.predict(X_test)
print(pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                    "Testing": get_classification_metrics(y_test, pred_test)}))

# ================================
# === MODEL 3: BASE + Feature Normalization ===
# ================================

print("\n=== MODEL 3: BASE + Feature Normalization ===")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

model3 = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                       gamma=1, reg_alpha=0.5, reg_lambda=1,
                       scale_pos_weight=scale_pos_weight,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")

model3.fit(X_train_scaled, y_train)
pred_train = model3.predict(X_train_scaled)
pred_test = model3.predict(X_test_scaled)
print(pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                    "Testing": get_classification_metrics(y_test, pred_test)}))

