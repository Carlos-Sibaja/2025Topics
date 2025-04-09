#M3_PredictionOverfitting2
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
df['RSI'] = df['RSI']
df['MACD'] = df['MACD']
df['MFI'] = df['MFI']

for col in ['Close', 'EMA_10', 'EMA_50', 'RSI', 'MACD', 'MFI']:
    df[f'{col}_lag1'] = df[col].shift(1)

# ===== Target = Predict only significant returns > 0.3% or < -0.3% =====
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
# === MODEL 1: BASE ===
# ================================

print("\n=== MODEL 1: BASE ===")
model1 = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
model1.fit(X_train, y_train)
pred_train = model1.predict(X_train)
pred_test = model1.predict(X_test)
print(pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                    "Testing": get_classification_metrics(y_test, pred_test)}))

# ================================
# === MODEL 2: BASE + REGULARIZATION ===
# ================================

print("\n=== MODEL 2: BASE + REGULARIZATION ===")
model2 = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                       gamma=1, reg_alpha=0.5, reg_lambda=1,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
model2.fit(X_train, y_train)
pred_train = model2.predict(X_train)
pred_test = model2.predict(X_test)
print(pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                    "Testing": get_classification_metrics(y_test, pred_test)}))

# ================================
# === MODEL 3: REG + TUNED N_ESTIMATORS ===
# ================================

print("\n=== MODEL 3: REGULARIZATION + TUNED ===")
model3 = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.05,
                       gamma=1, reg_alpha=0.5, reg_lambda=1,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
model3.fit(X_train, y_train)
pred_train = model3.predict(X_train)
pred_test = model3.predict(X_test)
print(pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                    "Testing": get_classification_metrics(y_test, pred_test)}))

# ================================
# === MODEL 4: REG + TUNING + NORMALIZATION ===
# ================================

print("\n=== MODEL 4: REG + TUNING + NORMALIZATION ===")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

model4 = XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.05,
                       gamma=1, reg_alpha=0.5, reg_lambda=1,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")

model4.fit(X_train_scaled, y_train)
pred_train = model4.predict(X_train_scaled)
pred_test = model4.predict(X_test_scaled)
print(pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                    "Testing": get_classification_metrics(y_test, pred_test)}))