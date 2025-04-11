# ===============================
# Import Libraries
# ===============================
import pandas as pd
from textblob import TextBlob
from datetime import timedelta

# ===============================
# Load NASDAQ Data
# ===============================
nasdaq = pd.read_csv('1M_nasdaq_data.csv', parse_dates=['Date'], index_col='Date')
nasdaq.index = pd.to_datetime(nasdaq.index, utc=True)
nasdaq.index = nasdaq.index.tz_convert(None)
nasdaq.index = nasdaq.index.normalize()  # Keep only date part

# ===============================
# Load News Content
# ===============================
news_df = pd.read_csv('2M_scraped_news.csv')

# change name of column published_date to date
news_df.rename(columns={'published_date': 'date'}, inplace=True)

news_df.index = pd.to_datetime(news_df['date'], utc=True)
news_df.index = news_df.index.tz_convert(None)
news_df.index = news_df.index.normalize()  # Keep only date part

# ===============================
# Move Saturday and Sunday news to Friday
# ===============================
for date in news_df.index:
    weekday = date.weekday()
    if weekday == 5:  # Saturday
        new_date = date - timedelta(days=1)
        news_df.loc[date, 'date'] = new_date
    elif weekday == 6:  # Sunday
        new_date = date - timedelta(days=2)
        news_df.loc[date, 'date'] = new_date

# ===============================
# Calculate Sentiment
# ===============================
def calculate_sentiment(text):
    if not isinstance(text, str) or text.strip() == '':
        return 0
    return TextBlob(text).sentiment.polarity

news_df['content_sentiment'] = news_df['description'].apply(calculate_sentiment)

# ===============================
# Aggregate Daily Sentiment
# ===============================
daily_sentiment = news_df.groupby(news_df.index).agg(
    Daily_Sentiment=('content_sentiment', 'mean'),
    Article_Count=('content_sentiment', 'count')
)

# ===============================
# Print Summary
# ===============================
print("\n📊 Daily Sentiment Summary (After Moving Weekend to Friday):")
print(daily_sentiment)

# ===============================
# Build Sentiment Features for NASDAQ Dates
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

print("\nSentiment features successfully appended to '3M_nasdaq_sentiment.csv'.")

