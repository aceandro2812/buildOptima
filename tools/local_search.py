# tools/local_search.py
"""
Local Web Search Tool - Free alternative to DuckDuckGo
Performs web crawling using Google search URLs with custom queries.
Similar to VS Code's approach where search terms are appended to base URLs.
"""

import requests
import time
import logging
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urljoin
from typing import List, Dict, Optional
from dataclasses import dataclass
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str

class LocalSearchTool:
    """
    Local search tool that crawls web results without API dependencies.
    Uses multiple search engines and implements respectful crawling practices.
    """
    
    def __init__(self, delay_range=(1, 3), max_results=5):
        self.delay_range = delay_range
        self.max_results = max_results
        self.session = self._create_session()
        
        # Search engine configurations
        self.search_engines = {
            'google': {
                'url': 'https://www.google.com/search?q={query}',
                'result_selector': 'div.g',
                'title_selector': 'h3',
                'link_selector': 'a',
                'snippet_selector': '.VwiC3b, .s3v9rd'
            },
            'bing': {
                'url': 'https://www.bing.com/search?q={query}',
                'result_selector': '.b_algo',
                'title_selector': 'h2 a',
                'link_selector': 'h2 a',
                'snippet_selector': '.b_caption p'
            }
        }
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
    
    def _create_session(self):
        """Create a requests session with retry strategy and proper headers."""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self):
        """Get randomized headers to avoid detection."""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def _respectful_delay(self):
        """Implement respectful crawling delay."""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\-.,()$%]', '', text)
        return text[:500]  # Limit length
    
    def _parse_google_results(self, soup: BeautifulSoup) -> List[SearchResult]:
        """Parse Google search results."""
        results = []
        config = self.search_engines['google']
        
        try:
            result_divs = soup.select(config['result_selector'])
            
            for div in result_divs[:self.max_results]:
                try:
                    # Extract title
                    title_elem = div.select_one(config['title_selector'])
                    title = self._clean_text(title_elem.get_text()) if title_elem else "No title"
                    
                    # Extract link
                    link_elem = div.select_one(config['link_selector'])
                    url = link_elem.get('href', '') if link_elem else ''
                    
                    # Clean Google redirect URLs
                    if url.startswith('/url?q='):
                        url = url.split('/url?q=')[1].split('&')[0]
                    
                    # Extract snippet
                    snippet_elem = div.select_one(config['snippet_selector'])
                    snippet = self._clean_text(snippet_elem.get_text()) if snippet_elem else "No description"
                    
                    if url and not url.startswith('http'):
                        continue  # Skip invalid URLs
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source='google'
                    ))
                    
                except Exception as e:
                    logging.warning(f"Error parsing individual Google result: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error parsing Google results: {e}")
        
        return results
    
    def _parse_bing_results(self, soup: BeautifulSoup) -> List[SearchResult]:
        """Parse Bing search results."""
        results = []
        config = self.search_engines['bing']
        
        try:
            result_divs = soup.select(config['result_selector'])
            
            for div in result_divs[:self.max_results]:
                try:
                    # Extract title and link (same element in Bing)
                    title_link_elem = div.select_one(config['title_selector'])
                    title = self._clean_text(title_link_elem.get_text()) if title_link_elem else "No title"
                    url = title_link_elem.get('href', '') if title_link_elem else ''
                    
                    # Extract snippet
                    snippet_elem = div.select_one(config['snippet_selector'])
                    snippet = self._clean_text(snippet_elem.get_text()) if snippet_elem else "No description"
                    
                    if not url or not url.startswith('http'):
                        continue  # Skip invalid URLs
                    
                    results.append(SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        source='bing'
                    ))
                    
                except Exception as e:
                    logging.warning(f"Error parsing individual Bing result: {e}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error parsing Bing results: {e}")
        
        return results
    
    def search(self, query: str, engine: str = 'google') -> List[SearchResult]:
        """
        Perform web search using specified search engine.
        
        Args:
            query: Search query string
            engine: Search engine to use ('google' or 'bing')
            
        Returns:
            List of SearchResult objects
        """
        if engine not in self.search_engines:
            raise ValueError(f"Unsupported search engine: {engine}")
        
        config = self.search_engines[engine]
        encoded_query = quote_plus(query)
        search_url = config['url'].format(query=encoded_query)
        
        logging.info(f"Searching {engine} for: {query}")
        
        try:
            # Respectful delay before request
            self._respectful_delay()
            
            # Make request with proper headers
            response = self.session.get(
                search_url,
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse results based on engine
            if engine == 'google':
                results = self._parse_google_results(soup)
            elif engine == 'bing':
                results = self._parse_bing_results(soup)
            else:
                results = []
            
            logging.info(f"Found {len(results)} results from {engine}")
            return results
            
        except requests.RequestException as e:
            logging.error(f"Request failed for {engine} search: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error during {engine} search: {e}")
            return []
    
    def multi_engine_search(self, query: str) -> List[SearchResult]:
        """
        Search using multiple engines and combine results.
        
        Args:
            query: Search query string
            
        Returns:
            Combined list of SearchResult objects from all engines
        """
        all_results = []
        
        for engine in ['google', 'bing']:
            try:
                results = self.search(query, engine)
                all_results.extend(results)
                
                # Add delay between different engines
                if engine != 'bing':  # Don't delay after last engine
                    self._respectful_delay()
                    
            except Exception as e:
                logging.error(f"Error searching {engine}: {e}")
                continue
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results[:self.max_results]
    
    def search_construction_materials(self, material_name: str, location: str = "India") -> str:
        """
        Specialized search for construction material pricing.
        
        Args:
            material_name: Name of the construction material
            location: Location for price context (default: India)
            
        Returns:
            Formatted string with search results
        """
        # Construct specialized query for construction materials
        query = f"{material_name} price rate {location} construction material supplier"
        
        try:
            results = self.multi_engine_search(query)
            
            if not results:
                return f"No search results found for {material_name} pricing information."
            
            # Format results for construction context
            formatted_results = f"Price research results for '{material_name}' in {location}:\n\n"
            
            for i, result in enumerate(results, 1):
                formatted_results += f"{i}. {result.title}\n"
                formatted_results += f"   Source: {result.source.title()}\n"
                formatted_results += f"   URL: {result.url}\n"
                formatted_results += f"   Info: {result.snippet}\n\n"
            
            return formatted_results
            
        except Exception as e:
            logging.error(f"Error in construction material search: {e}")
            return f"Error occurred while searching for {material_name} pricing: {str(e)}"

# Convenience function for easy integration
def search_material_prices(material_name: str, location: str = "India") -> str:
    """
    Quick function to search for construction material prices.
    
    Args:
        material_name: Name of the material to search for
        location: Location context for pricing
        
    Returns:
        Formatted search results string
    """
    search_tool = LocalSearchTool(max_results=3)
    return search_tool.search_construction_materials(material_name, location)

if __name__ == "__main__":
    # Test the search tool
    print("Testing Local Search Tool...")
    
    # Test basic search
    search_tool = LocalSearchTool(max_results=3)
    
    # Test construction material search
    test_material = "cement"
    print(f"\nTesting search for: {test_material}")
    results = search_tool.search_construction_materials(test_material, "Mumbai")
    print(results)