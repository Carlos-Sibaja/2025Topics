#M1_Sentimient Analysis
# ===============================
# M1_Sentiment_Analysis
# ===============================
# Import Libraries
# ===============================
import pandas as pd
import json
from textblob import TextBlob
from datetime import datetime, timedelta

# ===============================
# Load NASDAQ data
# ===============================
nasdaq = pd.read_csv('nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
nasdaq.index = pd.to_datetime(nasdaq.index, utc=True)
nasdaq.index = nasdaq.index.tz_convert(None)

# ===============================
# Load News Content (the full article text)
# ===============================
news_df = pd.read_csv('news_content.csv')

# Convert 'date' field to datetime
news_df['date'] = pd.to_datetime(news_df['date'], format='%d.%m.%y', errors='coerce')

# ===============================
# Extract Sentiment from News
# ===============================

# Function to calculate sentiment (-1 to 1)
def calculate_sentiment(text):
    if pd.isna(text) or text.strip() == '':
        return 0
    return TextBlob(text).sentiment.polarity

# Calculate sentiment based on the full 'content'
news_df['content_sentiment'] = news_df['content'].apply(calculate_sentiment)

# ===============================
# Daily Sentiment and Article Count
# ===============================

# Group by date and calculate daily average sentiment and number of articles
daily_sentiment = news_df.groupby('date').agg(
    Daily_Sentiment=('content_sentiment', 'mean'),
    Article_Count=('content_sentiment', 'count')
)

# ===============================
# Print Summary
# ===============================
print("\n📊 Daily Sentiment Summary:")
print(daily_sentiment)

# ===============================
# Build Sentiment Features for NASDAQ Dates
# ===============================
sentiment_features = pd.DataFrame(index=nasdaq.index)

for date in nasdaq.index:
    weekday = date.weekday()  # Monday = 0, Sunday = 6

    if weekday <= 3:  # Monday to Thursday
        t1 = date - timedelta(days=1)
        t2 = date - timedelta(days=2)
        t3 = date - timedelta(days=3)
    else:  # Friday (bundle Friday to Sunday news)
        t1 = date
        t2 = date - timedelta(days=1)
        t3 = date - timedelta(days=2)

    # Get sentiment values, use 0 if not available
    s_t1 = daily_sentiment.loc[daily_sentiment.index == t1.date(), 'Daily_Sentiment'].values
    s_t2 = daily_sentiment.loc[daily_sentiment.index == t2.date(), 'Daily_Sentiment'].values
    s_t3 = daily_sentiment.loc[daily_sentiment.index == t3.date(), 'Daily_Sentiment'].values

    s_t1 = s_t1[0] if len(s_t1) > 0 else 0
    s_t2 = s_t2[0] if len(s_t2) > 0 else 0
    s_t3 = s_t3[0] if len(s_t3) > 0 else 0

    # Store the sentiment features
    sentiment_features.loc[date, 'Sentiment_T1'] = s_t1
    sentiment_features.loc[date, 'Sentiment_T2'] = s_t2
    sentiment_features.loc[date, 'Sentiment_T3'] = s_t3
    sentiment_features.loc[date, 'Sentiment_3DayAVG'] = (s_t1 + s_t2 + s_t3) / 3

# ===============================
# Append Sentiment to NASDAQ
# ===============================
nasdaq = nasdaq.join(sentiment_features)

# ===============================
# Save Updated NASDAQ
# ===============================
nasdaq.to_csv('nasdaq.csv')
print("\n✅ Sentiment features successfully appended to 'nasdaq.csv'.")
print(nasdaq.head(-5))
