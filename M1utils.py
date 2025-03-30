#utils.py
# utils.py

from datetime import datetime

def scrape_site(playwright, source, site_info, day):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    locale="en-US",
    timezone_id="America/New_York",
    viewport={"width": 1280, "height": 800},
    java_script_enabled=True
)
    page = context.new_page()



    page.goto(site_info['url'], timeout=60000)
    # page.wait_for_timeout(10000) 
    page.wait_for_selector(site_info['selector'], timeout=15000)

    # try:
    #     # Espera explícita al selector
    #     page.wait_for_selector(site_info['selector'], timeout=10000)
    # except:
    #     print(f"⚠️ No se encontraron artículos en {source}")
    #     browser.close()
    #     return []

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


#  # Guardar el HTML para revisar
#     with open(f"reuters_debug_{day.strftime('%d_%m_%y')}.html", "w", encoding="utf-8") as f:
#         f.write(page.content())

#     print("✅ Página guardada como HTML para inspección manual.")
    browser.close()
    return news
