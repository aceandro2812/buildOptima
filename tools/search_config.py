# tools/search_config.py
"""
Configuration settings for the local search tool.
Customize these settings based on your needs and usage patterns.
"""

# Search behavior settings
DEFAULT_MAX_RESULTS = 5
DEFAULT_DELAY_RANGE = (1, 3)  # Seconds between requests (min, max)
DEFAULT_TIMEOUT = 10  # Request timeout in seconds

# Construction-specific search settings
CONSTRUCTION_LOCATIONS = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", 
    "Hyderabad", "Pune", "Ahmedabad", "Thane", "India"
]

# Search query templates for construction materials
MATERIAL_SEARCH_TEMPLATES = {
    'price': "{material} price rate {location} construction material supplier",
    'supplier': "{material} supplier {location} construction wholesale",
    'specification': "{material} specification grade quality {location}",
    'disposal': "{material} disposal recycling {location} waste management"
}

# User agents for rotation (helps avoid detection)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]

# Rate limiting settings
RATE_LIMIT_SETTINGS = {
    'requests_per_minute': 20,
    'burst_limit': 5,
    'cooldown_period': 60  # seconds
}

# Search engine configurations
SEARCH_ENGINES = {
    'google': {
        'enabled': True,
        'weight': 0.7,  # Preference weight
        'url_template': 'https://www.google.com/search?q={query}',
        'selectors': {
            'result_container': 'div.g',
            'title': 'h3',
            'link': 'a',
            'snippet': '.VwiC3b, .s3v9rd'
        }
    },
    'bing': {
        'enabled': True,
        'weight': 0.3,
        'url_template': 'https://www.bing.com/search?q={query}',
        'selectors': {
            'result_container': '.b_algo',
            'title': 'h2 a',
            'link': 'h2 a',
            'snippet': '.b_caption p'
        }
    }
}

# Content filtering settings
CONTENT_FILTERS = {
    'max_snippet_length': 500,
    'exclude_domains': [
        'facebook.com', 'twitter.com', 'instagram.com',
        'youtube.com', 'tiktok.com'
    ],
    'prefer_domains': [
        'indiamart.com', 'tradeindia.com', 'justdial.com',
        'sulekha.com', 'exportersindia.com'
    ]
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_search_queries': True,
    'log_results_count': True
}