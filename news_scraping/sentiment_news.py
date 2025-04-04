import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from collections import Counter


# Load the CSV file
df = pd.read_csv('scraped_news.csv')

# Convert the 'published_date' column to datetime
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

# Function to calculate sentiment
def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity  # Sentiment Score (-1 to 1)

# Apply sentiment calculation to the title and description
df['title_sentiment'] = df['title'].apply(calculate_sentiment)
df['description_sentiment'] = df['description'].apply(calculate_sentiment)

# Save the DataFrame with sentiment scores to a new CSV file
df.to_csv('scraped_news_with_sentiment.csv', index=False)
print("Sentiment scores calculated and saved to 'scraped_news_with_sentiment.csv'.")

# Function to plot sentiment distribution
def plot_sentiment_distribution(df, sentiment_column, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[sentiment_column], bins=30, kde=True)
    plt.title(title)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Plot sentiment distribution for title and description
plot_sentiment_distribution(df, 'title_sentiment', 'Sentiment Distribution of Titles')
plot_sentiment_distribution(df, 'description_sentiment', 'Sentiment Distribution of Descriptions')

# Function to plot sentiment over time
def plot_sentiment_over_time(df, sentiment_column, title):
    # Group by date and calculate the mean sentiment score
    df['date'] = df['published_date'].dt.date
    trend = df.groupby('date')[sentiment_column].mean()

    # Plot the trend
    plt.figure(figsize=(10, 5))
    plt.plot(trend.index, trend.values, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Plot sentiment over time for title and description
plot_sentiment_over_time(df, 'title_sentiment', 'Average Title Sentiment Over Time')
plot_sentiment_over_time(df, 'description_sentiment', 'Average Description Sentiment Over Time')

# Function to get the most common words in titles
def get_most_common_words(df, column, n=10):
    all_titles = " ".join(df[column])
    word_counts = Counter(all_titles.split())
    return word_counts.most_common(n)

# Get most common words in titles
common_words = get_most_common_words(df, 'title')
print("Most common words in titles:", common_words)

# Get most common words in descriptions
common_words_desc = get_most_common_words(df, 'description')
print("Most common words in descriptions:", common_words_desc)