# =============================================================================
# SEMANTIC SEARCH MODULE
# Embedding-based semantic search for improved RAG retrieval
# =============================================================================
"""
Semantic Search: Advanced Retrieval for Medical Knowledge.

WHAT IS THIS?
    This module implements semantic search using sentence embeddings
    to retrieve relevant medical knowledge based on meaning similarity
    rather than exact keyword matching.

WHY SEMANTIC SEARCH?
    Traditional keyword matching has limitations:
    1. Synonym blindness: "tumor" vs "neoplasm" vs "mass" won't match
    2. Context loss: Keywords ignore sentence meaning
    3. Vocabulary mismatch: Medical queries may use different terms
    
    Semantic search solves these by:
    1. Converting text to dense vector representations (embeddings)
    2. Using cosine similarity to find semantically similar content
    3. Understanding meaning, not just exact word matches

EMBEDDING MODEL:
    We use 'all-MiniLM-L6-v2' from sentence-transformers:
    - Compact (80MB) but effective
    - Optimized for semantic similarity
    - 384-dimensional embeddings
    - Works offline after first download

ARCHITECTURE:
    ┌─────────────┐      ┌──────────────┐      ┌─────────────┐
    │  Query Text │──────│  Embedding   │──────│  Query Vec  │
    └─────────────┘      │    Model     │      └──────┬──────┘
                         └──────────────┘             │
                                                      │ Cosine
    ┌─────────────┐      ┌──────────────┐             │ Similarity
    │  Knowledge  │──────│  Embedding   │             │
    │    Base     │      │    Model     │             │
    └─────────────┘      └──────────────┘             │
           │                                          │
           ▼                                          ▼
    ┌─────────────┐                          ┌─────────────┐
    │ Doc Vectors │─────────────────────────▶│  Top-K Docs │
    └─────────────┘                          └─────────────┘
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json
import pickle

# Check for sentence-transformers availability
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠ sentence-transformers not installed. Using fallback keyword matching.")
    print("  Install with: pip install sentence-transformers")


class SemanticSearch:
    """
    Semantic search engine using sentence embeddings.
    
    This class provides:
        - Text embedding using transformer models
        - Cosine similarity-based retrieval
        - Efficient vector indexing and search
        - Fallback to keyword matching if transformers unavailable
    
    Example:
        >>> search = SemanticSearch()
        >>> search.index_documents([
        ...     {"id": "1", "content": "Adenocarcinoma is a peripheral lung cancer"},
        ...     {"id": "2", "content": "Squamous cell carcinoma occurs centrally"}
        ... ])
        >>> results = search.search("peripheral tumor", top_k=1)
        >>> print(results[0]['content'])
        Adenocarcinoma is a peripheral lung cancer
    """
    
    # Default model - good balance of speed and quality
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    # Alternative models (uncomment to use)
    # "all-mpnet-base-v2"  # Higher quality, slower
    # "paraphrase-MiniLM-L6-v2"  # Good for paraphrase detection
    # "sentence-transformers/all-MiniLM-L12-v2"  # Slightly better, slightly slower
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the semantic search engine.
        
        Args:
            model_name: Name of the sentence-transformers model
            cache_dir: Directory to cache embeddings
            use_gpu: Whether to use GPU for embedding computation
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Document storage
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Initialize embedding model
        self.model = None
        self.use_semantic = SENTENCE_TRANSFORMERS_AVAILABLE
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                device = "cuda" if use_gpu else "cpu"
                self.model = SentenceTransformer(model_name, device=device)
                print(f"✓ Semantic search initialized with {model_name}")
            except Exception as e:
                print(f"⚠ Failed to load embedding model: {e}")
                print("  Falling back to keyword matching")
                self.use_semantic = False
        
        # Keyword index for fallback
        self.keyword_index: Dict[str, List[int]] = {}
    
    def _embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
        
        Returns:
            Numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model is None:
            raise RuntimeError("Embedding model not available")
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True  # For cosine similarity
        )
        
        return embeddings
    
    def _cosine_similarity(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Since embeddings are normalized, this is just dot product.
        
        Args:
            query_embedding: Query vector, shape (1, dim) or (dim,)
            doc_embeddings: Document vectors, shape (n_docs, dim)
        
        Returns:
            Similarity scores, shape (n_docs,)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Dot product of normalized vectors = cosine similarity
        similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
        
        return similarities
    
    def _build_keyword_index(self) -> None:
        """Build keyword index for fallback search."""
        self.keyword_index = {}
        
        for idx, doc in enumerate(self.documents):
            # Extract words from content
            content = doc.get('content', '') + ' ' + ' '.join(doc.get('keywords', []))
            words = content.lower().split()
            
            for word in words:
                # Clean word
                word = ''.join(c for c in word if c.isalnum())
                if len(word) >= 3:  # Skip short words
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    if idx not in self.keyword_index[word]:
                        self.keyword_index[word].append(idx)
    
    def _keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Fallback keyword-based search.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (doc_index, score) tuples
        """
        query_words = [
            ''.join(c for c in w if c.isalnum())
            for w in query.lower().split()
            if len(w) >= 3
        ]
        
        scores = {}
        for word in query_words:
            # Exact match
            if word in self.keyword_index:
                for idx in self.keyword_index[word]:
                    scores[idx] = scores.get(idx, 0) + 2
            
            # Partial match
            for key, indices in self.keyword_index.items():
                if word in key or key in word:
                    for idx in indices:
                        scores[idx] = scores.get(idx, 0) + 1
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Normalize scores
        max_score = ranked[0][1] if ranked else 1
        ranked = [(idx, score / max_score) for idx, score in ranked[:top_k]]
        
        return ranked
    
    def index_documents(
        self,
        documents: List[Dict],
        content_field: str = 'content',
        keywords_field: str = 'keywords'
    ) -> None:
        """
        Index documents for semantic search.
        
        Args:
            documents: List of document dictionaries
            content_field: Field name containing main text
            keywords_field: Field name containing keywords (optional)
        """
        self.documents = documents
        
        # Build keyword index (always, for fallback)
        self._build_keyword_index()
        
        if not self.use_semantic:
            print(f"✓ Indexed {len(documents)} documents (keyword mode)")
            return
        
        # Create combined text for each document
        texts = []
        for doc in documents:
            content = doc.get(content_field, '')
            keywords = doc.get(keywords_field, [])
            
            # Combine content and keywords for richer embeddings
            if isinstance(keywords, list):
                keywords_text = ' '.join(keywords)
            else:
                keywords_text = str(keywords)
            
            combined = f"{keywords_text} {content}"
            texts.append(combined)
        
        # Generate embeddings
        print(f"  Generating embeddings for {len(texts)} documents...")
        self.embeddings = self._embed(texts)
        
        print(f"✓ Indexed {len(documents)} documents (semantic mode)")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            threshold: Minimum similarity score (0-1)
        
        Returns:
            List of matching documents with similarity scores
        """
        if not self.documents:
            return []
        
        if not self.use_semantic or self.embeddings is None:
            # Fallback to keyword search
            ranked = self._keyword_search(query, top_k)
        else:
            # Semantic search
            query_embedding = self._embed(query)
            similarities = self._cosine_similarity(query_embedding, self.embeddings)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            ranked = [(idx, similarities[idx]) for idx in top_indices]
        
        # Build results
        results = []
        for idx, score in ranked:
            if score >= threshold:
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(score)
                results.append(doc)
        
        return results
    
    def save_index(self, filepath: str) -> None:
        """
        Save the index to disk.
        
        Args:
            filepath: Path to save the index
        """
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'model_name': self.model_name,
            'use_semantic': self.use_semantic
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Index saved to {filepath}")
    
    def load_index(self, filepath: str) -> bool:
        """
        Load an index from disk.
        
        Args:
            filepath: Path to the index file
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.embeddings = data.get('embeddings')
            self.use_semantic = data.get('use_semantic', False) and SENTENCE_TRANSFORMERS_AVAILABLE
            
            # Rebuild keyword index
            self._build_keyword_index()
            
            print(f"✓ Index loaded from {filepath} ({len(self.documents)} documents)")
            return True
            
        except Exception as e:
            print(f"⚠ Failed to load index: {e}")
            return False


class SemanticKnowledgeBase:
    """
    Medical knowledge base with semantic search capabilities.
    
    Enhances the basic MedicalKnowledgeBase with:
        - Semantic similarity matching
        - Hybrid search (semantic + keyword)
        - Query expansion using embeddings
    
    Example:
        >>> kb = SemanticKnowledgeBase()
        >>> kb.load_default_knowledge()
        >>> results = kb.search("peripheral ground glass opacity")
        >>> for r in results:
        ...     print(f"{r['similarity_score']:.2f}: {r['content'][:50]}...")
    """
    
    def __init__(
        self,
        model_name: str = SemanticSearch.DEFAULT_MODEL,
        use_gpu: bool = True
    ):
        """
        Initialize semantic knowledge base.
        
        Args:
            model_name: Sentence transformer model name
            use_gpu: Whether to use GPU acceleration
        """
        self.search_engine = SemanticSearch(
            model_name=model_name,
            use_gpu=use_gpu
        )
        self.entries: List[Dict] = []
        self._indexed = False
    
    def load_from_knowledge_base(self, kb) -> None:
        """
        Load entries from an existing MedicalKnowledgeBase.
        
        Args:
            kb: MedicalKnowledgeBase instance
        """
        self.entries = kb.entries.copy()
        self._reindex()
    
    def load_entries(self, entries: List[Dict]) -> None:
        """
        Load entries directly.
        
        Args:
            entries: List of knowledge entry dictionaries
        """
        self.entries = entries.copy()
        self._reindex()
    
    def _reindex(self) -> None:
        """Rebuild the search index."""
        self.search_engine.index_documents(
            self.entries,
            content_field='content',
            keywords_field='keywords'
        )
        self._indexed = True
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Dict]:
        """
        Search for relevant knowledge entries.
        
        Args:
            query: Search query
            top_k: Maximum number of results
            threshold: Minimum similarity score
        
        Returns:
            List of matching knowledge entries
        """
        if not self._indexed:
            print("⚠ Knowledge base not indexed. Call load_entries first.")
            return []
        
        return self.search_engine.search(query, top_k, threshold)
    
    def hybrid_search(
        self,
        query: str,
        class_name: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Hybrid search combining semantic and class-based filtering.
        
        Args:
            query: Search query
            class_name: Optional cancer class to prioritize
            top_k: Maximum number of results
        
        Returns:
            List of matching knowledge entries
        """
        # Get semantic results
        results = self.search(query, top_k=top_k * 2, threshold=0.2)
        
        if class_name:
            # Boost results matching the class
            class_lower = class_name.lower().replace('_', ' ')
            for result in results:
                keywords = [k.lower() for k in result.get('keywords', [])]
                content_lower = result.get('content', '').lower()
                
                if any(class_lower in k or k in class_lower for k in keywords):
                    result['similarity_score'] *= 1.5
                elif class_lower in content_lower:
                    result['similarity_score'] *= 1.2
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        return results[:top_k]


class SemanticPubMedSearch:
    """
    Semantic search wrapper for PubMed articles.
    
    Enhances PubMed retrieval with:
        - Semantic reranking of search results
        - Abstract similarity matching
        - Cross-article relevance scoring
    """
    
    def __init__(
        self,
        model_name: str = SemanticSearch.DEFAULT_MODEL,
        use_gpu: bool = True
    ):
        """
        Initialize semantic PubMed search.
        
        Args:
            model_name: Sentence transformer model name
            use_gpu: Whether to use GPU acceleration
        """
        self.search_engine = SemanticSearch(
            model_name=model_name,
            use_gpu=use_gpu
        )
        self.articles: List[Dict] = []
        self._indexed = False
    
    def index_articles(self, articles: List) -> None:
        """
        Index PubMed articles for semantic search.
        
        Args:
            articles: List of PubMedArticle objects or dicts
        """
        # Convert to dicts if needed
        self.articles = []
        for article in articles:
            if hasattr(article, 'to_dict'):
                self.articles.append(article.to_dict())
            elif isinstance(article, dict):
                self.articles.append(article)
        
        # Create searchable content
        docs = []
        for article in self.articles:
            doc = article.copy()
            doc['content'] = f"{article.get('title', '')} {article.get('abstract', '')}"
            docs.append(doc)
        
        self.search_engine.index_documents(docs, content_field='content')
        self._indexed = True
    
    def search(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Search indexed articles semantically.
        
        Args:
            query: Search query
            top_k: Maximum number of results
        
        Returns:
            List of matching articles with similarity scores
        """
        if not self._indexed:
            return []
        
        return self.search_engine.search(query, top_k)
    
    def rerank(
        self,
        query: str,
        articles: List,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Rerank a list of articles by semantic relevance.
        
        Args:
            query: Query to match against
            articles: List of articles to rerank
            top_k: Number of top results to return
        
        Returns:
            Reranked list of articles
        """
        # Index temporarily
        self.index_articles(articles)
        
        # Search and return
        return self.search(query, top_k)


def test_semantic_search():
    """Test semantic search functionality."""
    print("=" * 60)
    print("SEMANTIC SEARCH TEST")
    print("=" * 60)
    
    # Create test documents
    documents = [
        {
            "id": "1",
            "keywords": ["adenocarcinoma", "peripheral"],
            "content": "Adenocarcinoma typically presents in the peripheral regions of the lung, often in the outer third of the lung parenchyma."
        },
        {
            "id": "2", 
            "keywords": ["squamous", "central"],
            "content": "Squamous cell carcinoma typically arises in the central airways, near the hilum."
        },
        {
            "id": "3",
            "keywords": ["ground glass", "opacity"],
            "content": "Ground-glass opacity on CT imaging is frequently associated with adenocarcinoma."
        },
        {
            "id": "4",
            "keywords": ["cavitation", "necrosis"],
            "content": "Squamous cell carcinoma has the highest propensity for cavitation due to central tumor necrosis."
        },
        {
            "id": "5",
            "keywords": ["benign", "granuloma"],
            "content": "Granulomas are benign lesions that can mimic malignancy on imaging."
        }
    ]
    
    # Initialize search
    search = SemanticSearch()
    search.index_documents(documents)
    
    # Test queries
    test_queries = [
        "tumor in outer lung region",  # Should match adenocarcinoma
        "central airway cancer",  # Should match squamous
        "hazy lung opacity",  # Should match ground glass
        "non-malignant lung nodule",  # Should match benign
    ]
    
    print("\nSearch Results:")
    print("-" * 60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = search.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            score = result.get('similarity_score', 0)
            content = result['content'][:60] + "..."
            print(f"  {i}. [{score:.3f}] {content}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_semantic_search()
