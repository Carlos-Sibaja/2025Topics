# utils.py

# M1utils.py
# ===============================
# Utility Functions for Scraping
# ===============================

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

def scrape_site(playwright, source, site_info, day):
    """
    Scrape news articles from a specific news site for a given day.

    Args:
        playwright: Playwright instance.
        source (str): Name of the news source (e.g., "Google News").
        site_info (dict): Dictionary containing the URL and CSS selector.
        day (datetime): The date to associate with the news articles.

    Returns:
        list: List of dictionaries containing date, title, source, and link.
    """

    # Launch a headless browser
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    # Go to the specified URL
    try:
        page.goto(site_info['url'], timeout=60000)
    except PlaywrightTimeoutError:
        print(f"⚠ Timeout loading {site_info['url']}")
        browser.close()
        return []

    # Scroll down multiple times to load more news
    for _ in range(3):
        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(1500)

    # Wait for the news elements to appear
    try:
        page.wait_for_selector(site_info['selector'], timeout=10000)
    except PlaywrightTimeoutError:
        print(f"⚠ Selector not found for {source}")
        browser.close()
        return []

    # Collect news elements
    links = page.query_selector_all(site_info['selector'])
    news = []

    for link in links:
        title = link.get_attribute('aria-label') or link.inner_text().strip()
        href = link.get_attribute('href')

        if href and title:
            # If the link is relative, build the full link
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

