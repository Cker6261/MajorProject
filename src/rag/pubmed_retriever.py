# =============================================================================
# PUBMED RETRIEVER
# Fetches real medical literature from PubMed/NCBI API
# =============================================================================
"""
PubMed Retriever: Real Medical Literature Integration.

WHAT DOES THIS DO?
    Connects to the NCBI PubMed API to fetch:
    1. Relevant research articles based on query
    2. Article titles, abstracts, and metadata
    3. Real citations for evidence-based explanations

WHY PUBMED?
    - Gold standard for medical literature
    - Free API access (no authentication required for basic queries)
    - Provides credible, peer-reviewed sources
    - Enhances RAG with real medical knowledge

API ENDPOINTS:
    - ESearch: Search for article IDs matching a query
    - EFetch: Retrieve article details by ID
    - ESummary: Get article summaries

RATE LIMITS:
    - Without API key: 3 requests/second
    - With API key: 10 requests/second
    - We implement caching to minimize API calls
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PubMedArticle:
    """Represents a PubMed article."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    
    def to_citation(self) -> str:
        """Generate a citation string."""
        first_author = self.authors[0] if self.authors else "Unknown"
        if len(self.authors) > 1:
            author_str = f"{first_author} et al."
        else:
            author_str = first_author
        return f"{author_str}, {self.journal}, {self.pub_date}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'pmid': self.pmid,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'journal': self.journal,
            'pub_date': self.pub_date,
            'citation': self.to_citation()
        }


class PubMedRetriever:
    """
    Retrieves medical literature from PubMed.
    
    Features:
        - Searches PubMed for relevant articles
        - Caches results to minimize API calls
        - Extracts abstracts for RAG knowledge base
    
    Example:
        >>> retriever = PubMedRetriever()
        >>> articles = retriever.search("lung adenocarcinoma CT imaging")
        >>> for article in articles:
        ...     print(article.title)
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, cache_dir: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize PubMed retriever.
        
        Args:
            cache_dir: Directory for caching API responses
            api_key: Optional NCBI API key for higher rate limits
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./pubmed_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.34 if api_key else 0.5  # seconds
        
        # Prebuilt queries for lung cancer types
        self.cancer_queries = {
            'adenocarcinoma': 'lung adenocarcinoma CT imaging characteristics radiology',
            'squamous cell carcinoma': 'lung squamous cell carcinoma CT imaging central hilar',
            'squamous_cell_carcinoma': 'lung squamous cell carcinoma CT imaging central hilar',
            'large cell carcinoma': 'lung large cell carcinoma CT imaging peripheral mass',
            'large_cell_carcinoma': 'lung large cell carcinoma CT imaging peripheral mass',
            'normal': 'normal lung CT imaging anatomy',
            'normal cases': 'normal lung CT imaging anatomy',
            'benign': 'benign lung nodule CT imaging characteristics granuloma',
            'benign cases': 'benign lung nodule CT imaging characteristics granuloma'
        }
        
        print("✓ PubMed Retriever initialized")
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str) -> Optional[str]:
        """Make HTTP request with error handling."""
        try:
            self._rate_limit()
            
            # Add API key if available
            if self.api_key:
                separator = '&' if '?' in url else '?'
                url = f"{url}{separator}api_key={self.api_key}"
            
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'LungXAI-Research/1.0 (Academic Project)'}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.read().decode('utf-8')
                
        except urllib.error.URLError as e:
            print(f"⚠ Network error: {e}")
            return None
        except Exception as e:
            print(f"⚠ Request error: {e}")
            return None
    
    def _get_cache_path(self, query: str) -> Path:
        """Get cache file path for a query."""
        # Create safe filename from query
        safe_name = "".join(c if c.isalnum() else "_" for c in query[:50])
        return self.cache_dir / f"{safe_name}.json"
    
    def _load_from_cache(self, query: str) -> Optional[List[PubMedArticle]]:
        """Load cached results if available and not expired."""
        cache_path = self._get_cache_path(query)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if cache is still valid (7 days)
                cache_time = datetime.fromisoformat(data.get('timestamp', '2000-01-01'))
                if datetime.now() - cache_time < timedelta(days=7):
                    articles = []
                    for article_data in data.get('articles', []):
                        articles.append(PubMedArticle(**article_data))
                    return articles
            except Exception:
                pass
        
        return None
    
    def _save_to_cache(self, query: str, articles: List[PubMedArticle]):
        """Save results to cache."""
        cache_path = self._get_cache_path(query)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'articles': [a.to_dict() for a in articles]
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silently fail on cache write errors
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        use_cache: bool = True
    ) -> List[PubMedArticle]:
        """
        Search PubMed for articles matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of articles to return
            use_cache: Whether to use cached results
        
        Returns:
            List of PubMedArticle objects
        """
        # Check cache first
        if use_cache:
            cached = self._load_from_cache(query)
            if cached:
                print(f"  → Using cached results for: {query[:40]}...")
                return cached[:max_results]
        
        print(f"  → Searching PubMed for: {query[:40]}...")
        
        # Step 1: Search for article IDs
        search_url = (
            f"{self.BASE_URL}/esearch.fcgi?"
            f"db=pubmed&"
            f"term={urllib.parse.quote(query)}&"
            f"retmax={max_results}&"
            f"retmode=json&"
            f"sort=relevance"
        )
        
        search_response = self._make_request(search_url)
        if not search_response:
            return []
        
        try:
            search_data = json.loads(search_response)
            pmids = search_data.get('esearchresult', {}).get('idlist', [])
        except json.JSONDecodeError:
            return []
        
        if not pmids:
            return []
        
        # Step 2: Fetch article details
        articles = self._fetch_articles(pmids)
        
        # Cache the results
        if use_cache and articles:
            self._save_to_cache(query, articles)
        
        return articles
    
    def _fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch article details for given PMIDs."""
        if not pmids:
            return []
        
        # Use ESummary for basic info + abstracts
        pmid_str = ','.join(pmids)
        
        # Fetch summaries
        summary_url = (
            f"{self.BASE_URL}/esummary.fcgi?"
            f"db=pubmed&"
            f"id={pmid_str}&"
            f"retmode=json"
        )
        
        summary_response = self._make_request(summary_url)
        if not summary_response:
            return []
        
        # Fetch abstracts
        fetch_url = (
            f"{self.BASE_URL}/efetch.fcgi?"
            f"db=pubmed&"
            f"id={pmid_str}&"
            f"rettype=abstract&"
            f"retmode=text"
        )
        
        abstract_response = self._make_request(fetch_url)
        
        # Parse summaries
        articles = []
        try:
            summary_data = json.loads(summary_response)
            results = summary_data.get('result', {})
            
            for pmid in pmids:
                if pmid not in results:
                    continue
                
                article_data = results[pmid]
                
                # Extract authors
                authors = []
                for author in article_data.get('authors', [])[:3]:
                    authors.append(author.get('name', 'Unknown'))
                
                # Get publication date
                pub_date = article_data.get('pubdate', 'Unknown')
                if len(pub_date) > 10:
                    pub_date = pub_date[:10]
                
                articles.append(PubMedArticle(
                    pmid=pmid,
                    title=article_data.get('title', 'Unknown title'),
                    abstract=self._extract_abstract(abstract_response, pmid) if abstract_response else '',
                    authors=authors,
                    journal=article_data.get('source', 'Unknown journal'),
                    pub_date=pub_date
                ))
                
        except json.JSONDecodeError:
            pass
        
        return articles
    
    def _extract_abstract(self, abstract_text: str, pmid: str) -> str:
        """Extract abstract for a specific PMID from bulk text response."""
        # The text format has articles separated by blank lines
        # This is a simple extraction - could be improved
        if not abstract_text:
            return ""
        
        # Try to find abstract text (simplified extraction)
        lines = abstract_text.split('\n')
        abstract_parts = []
        in_abstract = False
        
        for line in lines:
            line = line.strip()
            if not line:
                if in_abstract and abstract_parts:
                    break
                continue
            
            # Skip headers and metadata
            if any(line.startswith(prefix) for prefix in ['PMID:', 'Author', 'Title:', 'DOI:', 'Copyright']):
                if in_abstract:
                    break
                continue
            
            # Start capturing after title-like content
            if len(line) > 50 and not line.endswith(':'):
                in_abstract = True
                abstract_parts.append(line)
            elif in_abstract:
                abstract_parts.append(line)
        
        abstract = ' '.join(abstract_parts)
        
        # Limit length
        if len(abstract) > 1000:
            abstract = abstract[:1000] + "..."
        
        return abstract
    
    def get_cancer_knowledge(self, cancer_type: str, max_articles: int = 3) -> List[Dict]:
        """
        Get PubMed knowledge for a specific cancer type.
        
        Args:
            cancer_type: Type of lung cancer
            max_articles: Maximum articles to retrieve
        
        Returns:
            List of knowledge entries formatted for RAG
        """
        # Get the appropriate query
        query = self.cancer_queries.get(
            cancer_type.lower(),
            f"lung {cancer_type} CT imaging characteristics"
        )
        
        articles = self.search(query, max_results=max_articles)
        
        # Convert to knowledge base format
        knowledge_entries = []
        for article in articles:
            if article.abstract:
                knowledge_entries.append({
                    'id': f"pubmed_{article.pmid}",
                    'keywords': cancer_type.lower().split() + ['pubmed', 'research'],
                    'content': article.abstract[:500] if len(article.abstract) > 500 else article.abstract,
                    'source': article.to_citation(),
                    'title': article.title,
                    'pmid': article.pmid,
                    'is_pubmed': True
                })
        
        return knowledge_entries
    
    def search_by_keywords(self, keywords: List[str], max_articles: int = 3) -> List[Dict]:
        """
        Search PubMed using a list of keywords.
        
        Args:
            keywords: List of search keywords
            max_articles: Maximum articles to retrieve
        
        Returns:
            List of knowledge entries
        """
        query = ' '.join(keywords) + ' lung cancer imaging'
        articles = self.search(query, max_results=max_articles)
        
        knowledge_entries = []
        for article in articles:
            if article.abstract:
                knowledge_entries.append({
                    'id': f"pubmed_{article.pmid}",
                    'keywords': keywords + ['pubmed'],
                    'content': article.abstract[:500],
                    'source': article.to_citation(),
                    'title': article.title,
                    'pmid': article.pmid,
                    'is_pubmed': True
                })
        
        return knowledge_entries


# Convenience function
def create_pubmed_retriever(cache_dir: Optional[str] = None) -> PubMedRetriever:
    """Create a PubMed retriever instance."""
    return PubMedRetriever(cache_dir=cache_dir)


if __name__ == "__main__":
    # Test the retriever
    retriever = PubMedRetriever()
    
    print("\nSearching for adenocarcinoma articles...")
    articles = retriever.search("lung adenocarcinoma CT imaging", max_results=2)
    
    for article in articles:
        print(f"\n{'='*60}")
        print(f"Title: {article.title}")
        print(f"Citation: {article.to_citation()}")
        print(f"Abstract: {article.abstract[:200]}...")
