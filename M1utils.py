# M1utils.py

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

def scrape_site(playwright, source, site_info, day):
    """
    Scrape news articles from a specific news site for a given day.
    """
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    try:
        page.goto(site_info['url'], timeout=6000)
    except PlaywrightTimeoutError:
        print(f"⚠ Timeout loading {site_info['url']}")
        browser.close()
        return []

    # Scroll down multiple times
    for _ in range(3):
        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(5000)

    try:
        page.wait_for_selector(site_info['selector'], timeout=5000)
    except PlaywrightTimeoutError:
        print(f"⚠ Selector not found for {source}")
        browser.close()
        return []

    links = page.query_selector_all(site_info['selector'])
    news = []

    for link in links:
        title = link.get_attribute('aria-label') or link.inner_text().strip()
        href = link.get_attribute('href')

        if href and title:
            if href.startswith('/'):
                href = site_info['url'].split('/search')[0] + href
            news.append({
                'date': day.strftime('%d.%m.%y'),
                'title': title,
                'source': source,
                'link': href
            })

    browser.close()
    return news