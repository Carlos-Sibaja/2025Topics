#M1_ScraperContent
# ===============================
# Import Libraries
# ===============================
import pandas as pd
import json
import requests
from newspaper import Article
from tqdm import tqdm

# ===============================
# Load News JSON
# ===============================
news_data = []
with open('trump_news_week.json', 'r') as f:
    for line in f:
        news_data.append(json.loads(line.strip()))

# Create a DataFrame
news_df = pd.DataFrame(news_data)

# ===============================
# Function to Download Article Content
# ===============================
def fetch_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

# ===============================
# Download Content for Each Link
# ===============================
# Create an empty list to store results
results = []

# tqdm for progress bar
for idx, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Fetching Articles"):
    date = row['date']
    link = row['link']
    
    content = fetch_article_text(link)
    
    results.append({
        'date': date,
        'link': link,
        'content': content
    })

# ===============================
# Save Results to CSV
# ===============================
results_df = pd.DataFrame(results)
results_df.to_csv('news_content.csv', index=False)
print("\nAll articles saved successfully to 'news_content.csv'.")