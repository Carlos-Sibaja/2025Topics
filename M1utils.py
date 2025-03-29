#utils.py
# utils.py

from datetime import datetime

def scrape_site(playwright, source, site_info, day):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(site_info['url'])
    page.wait_for_timeout(5000)  # Espera 5s

    news = []
    links = page.locator(site_info['selector']).all()

    for link in links:
        title = link.inner_text().strip()
        href = link.get_attribute('href')

        if href and title:
            if not href.startswith('http'):
                href = site_info['url'].split('/search')[0] + href
            news.append({
                'date': day.strftime('%d.%m.%y'),
                'title': title,
                'source': source,
                'link': href
            })

    browser.close()
    return news
