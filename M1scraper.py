# scraper.py

import random
import pandas as pd
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright
from M1config import news_sites
from M1utils import scrape_site

def main():
    all_news = []
    daily_summary = []

    today = datetime.now()
    days = 5 # Días a buscar

    with sync_playwright() as playwright:
        for i in range(days):
            day = today - timedelta(days=i)
            collected = []
            sources = list(news_sites.items())
            random.shuffle(sources)  # Aleatorizar orden de medios

            print(f"\n🔍 Searching news for {day.strftime('%d.%m.%y')}")

            for source, site_info in sources:
                if len(collected) >= 30:
                    break

                try:
                    news = scrape_site(playwright, source, site_info, day)
                    news_needed = 30 - len(collected)
                    news = news[:min(6,news_needed)]
                    collected.extend(news)
                    print(f"✅ {source}: {len(news)} articles collected")
                except Exception as e:
                    print(f"❌ Error in {source}: {e}")

            if len(collected) == 0:
                print(f"⚠ No news found for {day.strftime('%d.%m.%y')}")

            all_news.extend(collected)
            daily_summary.append({'date': day.strftime('%d.%m.%y'), 'count': len(collected)})

    # Guardar CSV
    df = pd.DataFrame(all_news)
    df.drop_duplicates(subset='link', inplace=True)
    df.to_json('trump_news_week.json', orient='records', lines=True)
    print("\n📄 JSON saved as trump_news_week.json")

    # Mostrar Resumen Diario
    df_summary = pd.DataFrame(daily_summary)
    print("\n📊 News Collection Summary:")
    print(df_summary)
    print("\nTotal articles collected:", len(all_news))

if __name__ == "__main__":
    main()
