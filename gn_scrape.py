# pip install fake-useragent

from GoogleNews import GoogleNews       # https://pypi.org/project/GoogleNews/  --  pip install GoogleNews
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
import random
from fake_useragent import UserAgent
import time

# Define search topics and sources
topics = ["Trump", "Stock Market"]
sources = [
    "bbc.com", "cnn.com", "nytimes.com", "reuters.com", "aljazeera.com",
    "bloomberg.com", "cnbc.com", "businessinsider.com",
    "techcrunch.com", "wired.com", "theverge.com",
    "nature.com", "sciencedaily.com", "cbc.ca", "theguardian.com", "reuters.com",
]


# Initialize GoogleNews
# googlenews = GoogleNews(
#     lang='en', 
#     encode='utf-8',
#     # period='30d'
#     start='01/01/2024',end='12/31/2024'     # custom day range (mm/dd/yyyy)
# )
googlenews = GoogleNews(encode='utf-8')
googlenews.set_lang('en')
googlenews.set_time_range('01/01/2024', '12/31/2024')  # Set the date range for the search

# Initialize UserAgent
ua = UserAgent()

# Store all news in a list
all_news = []

# Loop through topics and sources
for topic in topics:
    for source in sources:
        # Rotate User-Agent
        headers ={'User-Agent': ua.random}

        # Introduce a random delay to prevent rate limiting
        time.sleep(random.uniform(2, 5))

        # Perform the search
        googlenews.search(f"{topic} site:{source}")
        results = googlenews.results()
        
        # Append each result to our list
        for news in results:
            title = news["title"]
            sentiment_score = TextBlob(title).sentiment.polarity  # Sentiment Score (-1 to 1)
            sentiment = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

            all_news.append({
                "Topic": topic,
                "Title": title,
                "Date": news["date"],
                "Source": news["media"],
                "Sentiment": sentiment,
                "Sentiment Score": sentiment_score,
                "Link": news["link"]
            })

# Convert to a Pandas DataFrame
df = pd.DataFrame(all_news)

# Save to CSV (optional)
df.to_csv("./news_data.csv", index=False)

# See sentiment Distribution
print(df["Sentiment"].value_counts())

# Count how many articles each source has
print(df["Source"].value_counts())

# Combine all titles
all_titles = " ".join(df["Title"])

# Split words & count occurrences
word_counts = Counter(all_titles.split())

# Get most common words (excluding stopwords like "the", "is", etc)
common_words = word_counts.most_common(10)
print(common_words)

# # Convert date to Pandas datetime format
# df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# # Count articles per date
# trend = df.groupby(df["Date"].dt.date).count()

# # Plot trend
# plt.figure(figsize=(10, 5))
# plt.plot(trend.index, trend["Title"], marker="o", linestyle="-", color="b")
# plt.title("Number of News Articles Over Time")
# plt.xlabel("Date")
# plt.ylabel("Number of Articles")
# plt.grid(True)
# plt.show()

# Compare sentiment across sources
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="Source", hue="Sentiment", palette="coolwarm")
plt.title("Sentiment Analysis by News Source")
plt.xticks(rotation=45)
plt.show()

# # Sentiment over time
# df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# # Group by date and get average sentiment score
# sentiment_trend = df.groupby(df["Date"].dt.date)["Sentiment Score"].mean()

# # Plot sentiment trend
# plt.figure(figsize=(10, 5))
# plt.plot(sentiment_trend.index, sentiment_trend.values, marker="o", linestyle="-", color="g")
# plt.axhline(0, color="red", linestyle="--")  # Neutral line
# plt.title("Sentiment Trend Over Time")
# plt.xlabel("Date")
# plt.ylabel("Sentiment Score")
# plt.grid(True)
# plt.show()