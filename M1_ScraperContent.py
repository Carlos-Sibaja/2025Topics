# M1_ScraperContent
# M1_ScraperContent FINAL - Parallel + Clean + Auto-Save

# ===============================
# Import Libraries
# ===============================
import pandas as pd
from newspaper import Article
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re

# ===============================
# Load News CSV
# ===============================
# Load your file
news_df = pd.read_csv('news_trump_total.csv')

# Convert 'date' column
news_df['date'] = pd.to_datetime(news_df['date'], format='%d.%m.%y', errors='coerce')

# ===============================
# Filter only real articles
# ===============================
# Keep only links with /year/month/day/ structure
news_df = news_df[news_df['link'].str.contains(r'/\d{4}/\d{2}/\d{2}/', regex=True, na=False)]

print(f"✅ Found {len(news_df)} real article links after cleaning.")

# ===============================
# Function to Download Article Content
# ===============================
def fetch_article_text(row):
    url = row['link']
    date = row['date']
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        text = ""
    return {'date': date, 'link': url, 'content': text}

# ===============================
# Download Content in Parallel + Auto-Save
# ===============================
results = []
save_every = 1000   # Auto-save every 1000 articles
save_counter = 0
partial_file = 'news_content_partial.csv'

# Remove old partial file if it exists
if os.path.exists(partial_file):
    os.remove(partial_file)

# Create ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=20) as executor:  # 20 workers for your strong machine
    futures = [executor.submit(fetch_article_text, row) for idx, row in news_df.iterrows()]
    
    for idx, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Fetching Articles")):
        result = future.result()
        results.append(result)
        save_counter += 1
        
        # Auto-save every 1000 articles
        if save_counter >= save_every:
            partial_df = pd.DataFrame(results)
            partial_df.to_csv(partial_file, index=False)
            print(f"\n💾 Auto-saved {len(results)} articles to {partial_file}")
            save_counter = 0  # Reset counter

# ===============================
# Final Save
# ===============================
results_df = pd.DataFrame(results)
results_df.to_csv('news_content.csv', index=False)
print("\n✅ All articles saved successfully to 'news_content.csv'.")
