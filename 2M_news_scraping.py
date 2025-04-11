from gnews import GNews
import pandas as pd
import csv

# Initialize GNews with various parameters
google_news = GNews(
    language='en',
    country='US',
    period='none',
    start_date=(2025, 3, 1),
    end_date=(2025, 3, 30),
    max_results=100,
)

all_news = []

# Get news articles for the search query
results = google_news.get_news('Trump')

# Process each news item
for news in results:
    # Append extracted data to the list
    all_news.append({
        'title': news['title'],
        'description': news['description'],
        'url': news['url'],
        'published_date': news['published date'], 
        'source': news['publisher']
    })

# Convert to a Pandas DataFrame
df = pd.DataFrame(all_news)

with open('2M_news_scraping.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write new data row by row
    for index, row in df.iterrows():
        writer.writerow(row)

print("Data saved to gnews.csv")