# ===============================
# Import Libraries
# ===============================
import pandas as pd
import requests
from newspaper import Article, Config
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import re

# ===============================
# Setup File Paths
# ===============================
source_file = 'news_trump_totalCARLOS.csv'
output_reference = 'news_reference.csv'
scraped_output = 'scraped_content_temp.csv'
final_output = 'news_trump_totalCARLOS_V2.csv'

# ===============================
# Load and Prepare Dataset
# ===============================
print("📥 Loading source CSV...")
news_df = pd.read_csv(source_file)

# Add simple ID column
news_df['id'] = news_df.index  # Unique row ID

# ===============================
# Extract REAL Date from Title and Clean Title
# ===============================
def extract_date_from_title(title):
    if pd.isna(title):
        return None
    match = re.search(r'([A-Z][a-z]{2,8} \d{1,2}, \d{4})$', title.strip())
    if match:
        return match.group(1)
    else:
        return None

# Extract date string
news_df['date_str'] = news_df['title'].apply(extract_date_from_title)

# Parse extracted date string into real datetime
news_df['date'] = pd.to_datetime(news_df['date_str'], format='%b %d, %Y', errors='coerce')

# Clean the title: remove the date text at the end
news_df['title'] = news_df['title'].str.replace(r'([A-Z][a-z]{2,8} \d{1,2}, \d{4})$', '', regex=True).str.strip()

# Drop temporary column
news_df = news_df.drop(columns=['date_str'])

# ===============================
# 👀 Print first 3 rows and wait for user confirmation
print("\n👀 Preview the first 3 rows after adding ID and extracting real date:")
print(news_df[['id', 'date', 'title', 'source', 'link']].head(3))

input("\n⏸️ Press ENTER to continue...")

# ===============================
# Save reference with only ID, title, source, link
# ===============================
news_reference = news_df[['id', 'title', 'source', 'link']]
news_reference.to_csv(output_reference, index=False)
print(f"✅ Saved news_reference.csv with {len(news_reference)} entries (without date).")

# ===============================
# Smart Filter: Keep News, Drop Garbage
# ===============================
# Remove 'video', 'audio', 'gallery' links
bad_patterns = ['video', 'gallery', 'audio']
mask = news_df['link'].apply(lambda x: not any(bp in str(x).lower() for bp in bad_patterns))
news_df = news_df[mask]

# Save number of rows BEFORE dropping
before_drop = len(news_df)

# Drop rows where date is missing
news_df = news_df.dropna(subset=['date'])

# Save number of rows AFTER dropping
after_drop = len(news_df)

# Show how many were dropped
print(f"🧹 Dropped {before_drop - after_drop} rows with missing dates.")
print(f"✅ Remaining rows: {after_drop}")

print(f"✅ After filtering, {len(news_df)} links remain.")

# ===============================
# LIMIT to 1000 articles for testing
# ===============================
# 

# ===============================
# Initialize Content Column
# ===============================
news_df['content'] = ""

# Setup Newspaper3k config
user_config = Config()
user_config.request_timeout = 10  # 10 seconds timeout

# ===============================
# Scraper Function
# ===============================
def fetch_article_text(index, row):
    url = row['link']
    text = ""
    tries = 2

    for attempt in range(tries):
        try:
            article = Article(url, config=user_config)
            article.download()
            article.parse()
            text = article.text
            if text.strip():
                return index, text
        except Exception as e:
            print(f"❌ Newspaper3k attempt {attempt+1} failed for {url}: {e}")

    # Fallback with BeautifulSoup
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text() for p in paragraphs if p.get_text().strip())
    except Exception as e:
        print(f"❌ Fallback failed for {url}: {e}")
        text = ""

    return index, text

# ===============================
# Start Parallel Scraping
# ===============================
print("\n🚀 Starting article extraction...")
with ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(fetch_article_text, idx, row) for idx, row in news_df.iterrows()]
    results = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Articles"):
        results.append(future.result())

# ===============================
# Build Scraped DataFrame
# ===============================
scraped_df = pd.DataFrame(results, columns=['id', 'content'])

# Save scraped content temporarily
scraped_df.to_csv(scraped_output, index=False)
print(f"\n✅ Scraped content saved to {scraped_output}.")

# ===============================
# Merge Original Data (Add ID, Title, Source, Link)
# ===============================
news_reference = pd.read_csv(output_reference)

merged_df = pd.merge(scraped_df, news_reference, on='id', how='left')

# ===============================
# Fallback Processor: Title + Cleaned Link
# ===============================
def generate_fallback(row):
    if str(row['content']).strip():
        return row['content']  # Keep real content
    else:
        clean_link = str(row['link']).replace('https://', '').replace('http://', '').replace('www.', '')
        return f"{row['title']} {clean_link}"

merged_df['content'] = merged_df.apply(generate_fallback, axis=1)

# ===============================
# Merge Date back
# ===============================
# Merge the real date again using id
final_merge = pd.merge(merged_df, news_df[['id', 'date']], on='id', how='left')

# ===============================
# Save Final Output
# ===============================
final_df = final_merge[['id', 'date', 'title', 'source', 'link', 'content']]

# Sort by ID
final_df = final_df.sort_values('id')

final_df.to_csv(final_output, index=False)
print(f"\n✅ Final CSV saved to {final_output}")

# ===============================
# Final Report
# ===============================
total_articles = len(final_df)
fallback_used = (scraped_df['content'].apply(lambda x: not str(x).strip())).sum()

print("\n📊 Final Report:")
print(f"Total articles processed: {total_articles}")
print(f"Articles with full scraped content: {total_articles - fallback_used}")
print(f"Articles with fallback (title + link): {fallback_used}")

