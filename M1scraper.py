# scraper.py

import random
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from playwright.sync_api import sync_playwright
from M1config import news_sites
from M1utils import scrape_site

def main():
    all_news = []
    daily_summary = []

    # ===============================
    # Set the fixed start date (March 31, 2025)
    # ===============================
    start_date = datetime(2025, 3, 31)
    days = 2 # Number of days to go backwards
    target_per_day = 76  # Target number of articles per day
    target_per_source = 10  # 🎯 Max articles per source per day

    with sync_playwright() as playwright:
        for i in range(days):
            day = start_date - timedelta(days=i)
            collected = []
            source_counts = defaultdict(int)  # 🧠 Track how many articles per source
            sources = list(news_sites.items())
            random.shuffle(sources)

            print(f"\n🔍 Searching news for {day.strftime('%d.%m.%y')}")

            for source, site_info in sources:
                if len(collected) >= target_per_day:
                    print(f"✅ Reached {target_per_day} articles for {day.strftime('%d.%m.%y')}. Moving to next day.")
                    break  # 🚀 Total target reached

                if source_counts[source] >= target_per_source:
                    print(f"⚠️ Source limit reached for {source} ({target_per_source} articles)")
                    continue  # 🚫 Skip if this source already reached its daily limit

                try:
                    news = scrape_site(playwright, source, site_info, day)
                    remaining_day = target_per_day - len(collected)
                    remaining_source = target_per_source - source_counts[source]
                    fetch_limit = min(len(news), remaining_day, remaining_source)

                    news = news[:fetch_limit]
                    collected.extend(news)
                    source_counts[source] += len(news)

                    print(f"✅ {source}: {len(news)} articles collected (Total: {len(collected)})")
                except Exception as e:
                    print(f"❌ Error in {source}: {e}")

            if len(collected) == 0:
                print(f"⚠️ No news found for {day.strftime('%d.%m.%y')}")

            all_news.extend(collected)
            daily_summary.append({'date': day.strftime('%d.%m.%y'), 'count': len(collected)})

    # ===============================
    # Save News Dataset
    # ===============================

    df = pd.DataFrame(all_news)
    df.to_csv('trump_news_week.csv', index=False)
    print("\n📄 CSV saved as trump_news_week.csv")

    # ===============================
    # Prepare and Print Daily Summary
    # ===============================

    df_summary = pd.DataFrame(daily_summary)
    df_summary['date'] = pd.to_datetime(df_summary['date'], format='%d.%m.%y', errors='coerce')
    df_summary = df_summary.sort_values('date').reset_index(drop=True)
    df_summary = df_summary.drop_duplicates(subset='date')

    print("\n📊 News Collection Summary (sorted):")
    print(df_summary)

    df_summary.to_csv('daily_news_summary.csv', index=False)

    print("\nTotal articles collected:", len(df))

if __name__ == "__main__":
    main()