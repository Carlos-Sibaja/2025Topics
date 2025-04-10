import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('scraped_news.csv')

# Convert the 'published_date' column to datetime
df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

# plot the number of articles over time
def plot_articles_over_time(df):
    # Group by date and count the number of articles
    df['date'] = df['published_date'].dt.date
    trend = df.groupby('date').size()

    # Plot the trend
    plt.figure(figsize=(10, 5))
    plt.plot(trend.index, trend.values, marker='o', linestyle='-', color='b')
    plt.title('Number of News Articles Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Call the function to plot
plot_articles_over_time(df)
