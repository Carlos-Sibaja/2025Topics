# deactivate to deactivate the virtual environment  .venv\Scripts\deactivate
# .venv\Scripts\Activate.ps1 to activate the virtual environment
# python -m venv .venv to create the virtual environment
# pip install -r requirements.txt to install the required packages
# pip freeze > requirements.txt to save the installed packages
# pip install scraper
# pip python d_elpais.py
#pip install streamlit



# config.py

news_sites = {
    'BBC': {
        'url': 'https://www.bbc.co.uk/search?q=trump',
        'selector': '.ssrcss-1yagzb7-PromoLink'
    },
    'The New York Times': {
        'url': 'https://www.nytimes.com/search?query=trump',
        'selector': '.css-1l4w6pd a'
    },
    'NPR': {
        'url': 'https://www.npr.org/search?query=trump',
        'selector': '.title a'
    },
    'CBS News': {
        'url': 'https://www.cbsnews.com/search/?q=trump',
        'selector': '.item__anchor'
    },
    'ABC News': {
        'url': 'https://abcnews.go.com/search?searchtext=trump',
        'selector': '.ContentRoll__Headline a'
    },   
    'Financial Times': {
        'url': 'https://www.ft.com/search?q=trump',
        'selector': '.o-teaser__heading a'
    },    
    'Politico': {
    'url': 'https://www.politico.com/search?q=trump',
    'selector': 'article.story-frag a'
  },
  
    'NBC': {
        'url': 'https://www.nbcnews.com/search?q=trump',
        'selector': '.queryly_item_row a'
    },
   'Al Jazeera': {
        'url': 'https://www.aljazeera.com/search/trump',
        'selector': '.u-clickable-card__link'
    },
    'FOX News': {
        'url': 'https://www.foxnews.com/search-results/search#q=trump',
        'selector': '.title a'
    }

    #  Estás siendo bloqueado por un sistema anti-bot llamado DataDome
#    'Reuters': {
#     'url': 'https://www.reuters.com/site-search/?query=trump',
#     'selector': 'a[data-testid="Heading"]'
#     },
    # PAY SITE
    # 'Bloomberg': {
    #     'url': 'https://www.bloomberg.com/search?query=trump',
    #     'selector': 'div.storyItem a'
    #},
    # Subscription
    #    'The Times (UK)': {
    #     'url': 'https://www.thetimes.co.uk/search?query=trump',
    #     'selector': '.Item-headline a'
    # },
#   'Associated Press (AP)': {
#         'url': 'https://apnews.com/search?q=trump',
#         'selector': '.PagePromo a.Link'
#     },
    #  'AFP': {
    #     'url': 'https://www.afp.com/en/search/trump',
    #     'selector': '.afp-hm-link'
    # }
}
