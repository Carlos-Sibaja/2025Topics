#  Using News API (newsapi.org)
#  https://newsapi.org/docs/getting-started
#  pip install newsapi-python


from newsapi import NewsApiClient
API_KEY = 'ecdba5bfdee147109099cc782017da1a'

# init
newsapi = NewsApiClient(api_key=API_KEY)

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(q='trump',        # Keyword of phrases to search for
                                            sources='bbc-news, cnn, apnews',    # Comma-separated string of identifiers (see /v2/sources)
                                            #from_param='2023-10-01',  # From date (YYYY-MM-DD)
                                            language='en',
                                            page_size=100,  # Number of results to return per page (default: 20)
                                            page=1) # Page number to retrieve (default: 1)
# /v2/everything
all_articles = newsapi.get_everything(q='trump',
                                        language='en',
                                        sort_by='relevancy',
                                        page_size=100,
                                        page=1)
# /v2/top-headlines/sources
sources = newsapi.get_sources()

# show results
print("Top Headlines:")
for article in top_headlines['articles']:
    print(article['title'])
    print(article['url'])
    print(article['publishedAt'])
    print("-----")
print("All Articles:")
for article in all_articles['articles']:
    print(article['title'])
    print(article['url'])
    print(article['publishedAt'])
    print("-----")
# print("Sources:")
# for source in sources['sources']:
#     print(source['name'])
#     print(source['id'])
#     print(source['description'])
#     print("-----")

