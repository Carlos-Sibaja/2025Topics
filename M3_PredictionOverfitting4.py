#M3_PredictionOverfitting4

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ===== Load Dataset =====
data = pd.read_csv('nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
data.index = pd.to_datetime(data.index, utc=True)
data.index = data.index.tz_convert(None)
data = data.drop(columns=['Close_Predicted_RF', 'Close_Predicted_XGB'], errors='ignore')
print("===== Dataset Loaded =====")

# ===== Feature Engineering (selected only) =====
df = data.copy()
df['Return'] = df['Close'].pct_change()
df['Volatility'] = df['Return'].rolling(5).std()

for col in ['Close', 'RSI', 'MACD', 'MFI']:
    df[f'{col}_lag1'] = df[col].shift(1)
    df[f'{col}_lag2'] = df[col].shift(2)

# ===== Target: Significant returns only =====
df['Rolling_Return'] = df['Return'].rolling(3).sum().shift(-3)
df['Target'] = np.where(df['Rolling_Return'] > 0.003, 1,
                        np.where(df['Rolling_Return'] < -0.003, 0, np.nan))
df.dropna(inplace=True)
print(f"Target distribution:\n{df['Target'].value_counts(normalize=True) * 100}")

features = [col for col in df.columns if 'lag' in col] + ['Return', 'Volatility']

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

# =============== MODEL 1: BASE =================
print("\n=== MODEL 1: BASE (Original Model 2) ===")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
model1 = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                       gamma=1, reg_alpha=0.5, reg_lambda=1,
                       scale_pos_weight=scale_pos_weight,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
model1.fit(X_train, y_train)
pred_train = model1.predict(X_train)
pred_test = model1.predict(X_test)
result1 = pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                        "Testing": get_classification_metrics(y_test, pred_test)})
print(result1)

# =============== MODEL 2: BASE + Hyperparameter Tuning =============
print("\n=== MODEL 2: BASE + Tuned ===")
model2 = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03,
                       gamma=2, reg_alpha=1, reg_lambda=1.5,
                       scale_pos_weight=scale_pos_weight,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
model2.fit(X_train, y_train)
pred_train = model2.predict(X_train)
pred_test = model2.predict(X_test)
result2 = pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                        "Testing": get_classification_metrics(y_test, pred_test)})
print(result2)

# =============== MODEL 3: BASE + Feature Selection Only =============
print("\n=== MODEL 3: BASE + Feature Selection ===")
selected_features = [f for f in features if 'MACD' not in f and 'MFI' not in f]  # example filter
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

model3 = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                       gamma=1, reg_alpha=0.5, reg_lambda=1,
                       scale_pos_weight=scale_pos_weight,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
model3.fit(X_train_sel, y_train)
pred_train = model3.predict(X_train_sel)
pred_test = model3.predict(X_test_sel)
result3 = pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                        "Testing": get_classification_metrics(y_test, pred_test)})
print(result3)

# =============== MODEL 4: BASE + Tuned + Feature Selection =============
print("\n=== MODEL 4: BASE + Tuned + Feature Selection ===")
model4 = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03,
                       gamma=2, reg_alpha=1, reg_lambda=1.5,
                       scale_pos_weight=scale_pos_weight,
                       random_state=42, verbosity=0, use_label_encoder=False, eval_metric="logloss")
model4.fit(X_train_sel, y_train)
pred_train = model4.predict(X_train_sel)
pred_test = model4.predict(X_test_sel)
result4 = pd.DataFrame({"Training": get_classification_metrics(y_train, pred_train),
                        "Testing": get_classification_metrics(y_test, pred_test)})
print(result4)

# =================== COMPARISON ===================
print("\n=== MODEL COMPARISON ===")
comparison = pd.concat([result1['Testing'], result2['Testing'], result3['Testing'], result4['Testing']], axis=1)
comparison.columns = ['Model1_Base', 'Model2_Tuned', 'Model3_SelectedFeatures', 'Model4_Tuned+Features']
print(comparison)