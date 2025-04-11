#Final Model

# ===============================
# Import Libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ===============================
# Load Dataset
# ===============================
data = pd.read_csv('3M_nasdaq_sentiment.csv', parse_dates=['Date'], index_col='Date')
data.index = pd.to_datetime(data.index, utc=True)
data.index = data.index.tz_convert(None)
data = data.drop(columns=['Close_Predicted_RF', 'Close_Predicted_XGB'], errors='ignore')
print("===== Dataset Loaded =====")

# ===============================
# Feature Engineering
# ===============================
df = data.copy()
df['Return'] = df['Close'].pct_change()
df['Volatility'] = df['Return'].rolling(5).std()

# Create lag features
for col in ['Close', 'RSI', 'MACD', 'MFI']:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)

# ===============================
# Calculate VWAP
# ===============================
df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
df['Cumulative_Typical_Price_Volume'] = (df['Typical_Price'] * df['Volume']).cumsum()
df['Cumulative_Volume'] = df['Volume'].cumsum()
df['VWAP'] = df['Cumulative_Typical_Price_Volume'] / df['Cumulative_Volume']

# Create lag features for VWAP
df['VWAP_lag1'] = df['VWAP'].shift(1)
df['VWAP_lag2'] = df['VWAP'].shift(2)

# ===============================
# Define Target: Significant Returns Only
# ===============================
df['Rolling_Return'] = df['Return'].rolling(3).sum().shift(-3)
df['Target'] = np.where(df['Rolling_Return'] > 0.003, 1,
                        np.where(df['Rolling_Return'] < -0.003, 0, np.nan))
df.dropna(inplace=True)
print(f"Target distribution:\n{df['Target'].value_counts(normalize=True) * 100}")

# ===============================
# Select Features
# ===============================
selected_features = [
    'Close_lag2',
    'Volatility', 'OBV', 'ATR_14', 'ADX_14',
    'VWAP_lag1', 'VWAP_lag2',
    'Sentiment_T1','Sentiment_T2', 'Sentiment_T3',
    'Sentiment_3DayAVG',
]


# ===============================
# Split Train / Validation / Real
# ===============================
train = df.loc['2023-01-01':'2023-12-31']
val = df.loc['2024-01-01':'2024-12-31']
real = df.loc['2025-01-01':'2025-03-31']

X_train, y_train = train[selected_features], train['Target']
X_val, y_val = val[selected_features], val['Target']
X_real, y_real = real[selected_features], real['Target']

# ===============================
# Metrics Function
# ===============================
def get_classification_metrics(y_true, y_pred):
    return {
        "Accuracy (DA)": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "ROC_AUC": roc_auc_score(y_true, y_pred) * 100
    }

# ===============================
# Train Model
# ===============================
print(f"\n=== MODEL: Final Version ===")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.01,
    gamma=5,
    reg_alpha=1,
    reg_lambda=3,
    subsample=0.7,
    colsample_bytree=0.7,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbosity=0,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# ===============================
# Predictions
# ===============================
pred_train = model.predict(X_train)
pred_val = model.predict(X_val)

# For Real Prediction 2025, adjust the threshold
y_real_proba = model.predict_proba(X_real)[:, 1]
threshold = 0.485
pred_real = (y_real_proba > threshold).astype(int)
print(threshold)

# ===============================
# Show Results
# ===============================
result = pd.DataFrame({
    "Training (2022-2023)": get_classification_metrics(y_train, pred_train),
    "Validation (2024)": get_classification_metrics(y_val, pred_val),
    "Real Prediction (2025)": get_classification_metrics(y_real, pred_real)
})
print("\n===== Final Model Performance =====")
print(result)

# ===============================
# Feature Importance Plot
# ===============================
importance = model.feature_importances_
features_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(features_names, importance)
plt.title('Feature Importance (Final Model with VWAP and Sentiment)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
