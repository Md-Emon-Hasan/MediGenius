import os

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

load_dotenv()

# Global instances
_wiki_wrapper = None
_tavily_search = None


def get_wikipedia_wrapper():
    global _wiki_wrapper
    if _wiki_wrapper is None:
        _wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=2000,
            load_all_available_meta=True
        )
    return _wiki_wrapper


def get_tavily_search():
    global _tavily_search
    if _tavily_search is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("TAVILY_API_KEY not found")
            return None
        _tavily_search = TavilySearchResults(api_key=api_key, max_results=3)
    return _tavily_search
