#Sentimient Analysis

# ===============================
# Import Libraries
# ===============================
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ===============================
# Load the Enhanced NASDAQ Dataset
# ===============================
nasdaq = pd.read_csv('M1_nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
nasdaq.index = pd.to_datetime(nasdaq.index, utc=True)
nasdaq.index = nasdaq.index.tz_convert(None)
nasdaq.index = nasdaq.index.normalize()  # Keep only date part

# ===============================
# Create Target Variable: Next Day's Closing Price
# ===============================
df['Target_Close'] = df['Close'].shift(-1)
df.dropna(inplace=True)  # Drop last row with NaN target

# ===============================
# Feature Selection
# ===============================
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'Sentiment_T1', 'Sentiment_T2', 'Sentiment_T3', 'Sentiment_3DayAVG']

X = df[features]
y = df['Target_Close']

# ===============================
# Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ===============================
# Train XGBoost Regressor
# ===============================
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ===============================
# Predict and Evaluate
# ===============================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📈 Mean Squared Error: {mse:.4f}")
print(f"🧮 R^2 Score: {r2:.4f}")

# ===============================
# Plot Actual vs Predicted
# ===============================
sentiment_features = pd.DataFrame(index=nasdaq.index)

for date in nasdaq.index:
    if date >= pd.Timestamp('2023-01-01'):

        weekday = date.weekday()  # Monday = 0, Sunday = 6

        if weekday <= 3:  # Monday to Thursday
            t1 = date - timedelta(days=1)
            t2 = date - timedelta(days=2)
            t3 = date - timedelta(days=3)
        else:  # Friday
            t1 = date
            t2 = date - timedelta(days=1)
            t3 = date - timedelta(days=2)

        # Lookup sentiments safely
        s_t1 = daily_sentiment.loc[daily_sentiment.index == t1, 'Daily_Sentiment'].values
        s_t2 = daily_sentiment.loc[daily_sentiment.index == t2, 'Daily_Sentiment'].values
        s_t3 = daily_sentiment.loc[daily_sentiment.index == t3, 'Daily_Sentiment'].values

        s_t1 = s_t1[0] if len(s_t1) > 0 else 0
        s_t2 = s_t2[0] if len(s_t2) > 0 else 0
        s_t3 = s_t3[0] if len(s_t3) > 0 else 0

        # Save features
        sentiment_features.loc[date, 'Sentiment_T1'] = s_t1
        sentiment_features.loc[date, 'Sentiment_T2'] = s_t2
        sentiment_features.loc[date, 'Sentiment_T3'] = s_t3
        sentiment_features.loc[date, 'Sentiment_3DayAVG'] = (s_t1 + s_t2 + s_t3) / 3

# ===============================
# Append Sentiment to NASDAQ
# ===============================
nasdaq = nasdaq.join(sentiment_features, rsuffix='_new')

# ===============================
# Save Updated NASDAQ
# ===============================
nasdaq.to_csv('3M_nasdaq_sentiment.csv')

print("\n✅ Sentiment features successfully appended to 'nasdaq_roberto.csv'.")
print(nasdaq.tail(10))
