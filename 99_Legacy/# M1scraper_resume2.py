# M1scraper_resume.py

import random
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from playwright.sync_api import sync_playwright
from M1config import news_sites
from M1utils import scrape_site
import os

# ✅ Define meses permitidos
#allowed_months = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # FEB to DEC
allowed_days = [1,2,3,5, 6, 7, 8, 9, 10, 11, 12,13,20]

def main():
    all_news = []
    daily_summary = []

    # ===============================
    # Set Parameters
    # ===============================
    start_date = datetime(2024, 4, 1)   # 🛠️ Initial start date
    end_date = datetime(2024, 4, 20)   # 🛠️ Final end date (adjust if needed)
    target_per_day = 33
    target_per_source = 12

    # ===============================
    # Check Resume Point
    # ===============================
    resume_from_date = start_date
    if os.path.exists('daily_news_summary.csv'):
        try:
            df_summary = pd.read_csv('daily_news_summary.csv')
            df_summary['date'] = pd.to_datetime(df_summary['date'], errors='coerce')
            last_scraped_date = df_summary['date'].max()
            resume_from_date = last_scraped_date + timedelta(days=1)
            print(f"🔄 Resuming from {resume_from_date.strftime('%d.%m.%Y')}")
        except Exception as e:
            print(f"⚠️ Warning: Could not read resume file, starting fresh. {e}")
    else:
        print(f"🆕 No previous summary found. Starting fresh from {start_date.strftime('%d.%m.%Y')}")

    # ===============================
    # Generate list of specific days (4th of each month)
    # ===============================
    special_days = pd.date_range(start=start_date, end=end_date, freq='D')
    special_days = [d for d in special_days if (d.day in allowed_days)]

    print(f"📅 Total days to scrape: {len(special_days)}")

    # ===============================
    # Start Scraping
    # ===============================
    with sync_playwright() as playwright:
        for day in special_days:
            collected = []
            source_counts = defaultdict(int)
            sources = list(news_sites.items())
            random.shuffle(sources)

            print(f"\n🔍 Searching news for {day.strftime('%d.%m.%y')}")

            for source, site_info in sources:
                if len(collected) >= target_per_day:
                    print(f"✅ Reached {target_per_day} articles for {day.strftime('%d.%m.%y')}. Moving to next day.")
                    break

                if source_counts[source] >= target_per_source:
                    continue

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

            # Save day result
            df_day = pd.DataFrame(collected)
            if not df_day.empty:
                if os.path.exists('trump_news_week.csv'):
                    df_day.to_csv('trump_news_week.csv', mode='a', index=False, header=False)
                else:
                    df_day.to_csv('trump_news_week.csv', index=False)

            # Update daily summary
            daily_summary.append({'date': day.strftime('%d.%m.%Y'), 'count': len(collected)})
            df_summary = pd.DataFrame(daily_summary)
            df_summary['date'] = pd.to_datetime(df_summary['date'], format='%d.%m.%Y', errors='coerce')
            df_summary = df_summary.sort_values('date').drop_duplicates(subset='date')
            df_summary.to_csv('daily_news_summary.csv', index=False)

    print("\n✅ Scraping completed!")
    print("\nTotal articles collected:", sum([d['count'] for d in daily_summary]))

if __name__ == "__main__":
    main()
