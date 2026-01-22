# =============================================================================
# RAG MODULE (RETRIEVAL-AUGMENTED GENERATION)
# Converts XAI visual evidence to textual medical explanations
# =============================================================================
"""
RAG Module for Explainable Lung Cancer Classification.

This module contains:
    - MedicalKnowledgeBase: Local JSON-based medical knowledge store
    - SemanticKnowledgeBase: Embedding-based semantic search for knowledge
    - SemanticSearch: Core semantic similarity search engine
    - XAITextConverter: Converts Grad-CAM regions to textual descriptions
    - ExplanationGenerator: Creates evidence-based explanations
    - PubMedRetriever: Fetches real medical literature from PubMed/NCBI

WHY RAG FOR THIS PROJECT?
    - Bridges the gap between visual XAI (Grad-CAM) and human understanding
    - Makes predictions more interpretable for non-technical users
    - Adds clinical context to model decisions
    - Provides evidence-based explanations with real citations

KNOWLEDGE SOURCES:
    1. Local Knowledge Base: Curated medical facts (fast, always available)
    2. PubMed API: Real research articles (authoritative, up-to-date)
    3. Semantic Search: Embedding-based retrieval for better matching

SEMANTIC SEARCH ADVANTAGE:
    Traditional keyword matching fails when:
    - "tumor" vs "neoplasm" vs "mass" - same meaning, different words
    - "peripheral lung region" vs "outer parenchyma" - semantic equivalence
    
    Semantic search uses sentence embeddings to understand MEANING,
    not just exact word matches. This dramatically improves retrieval quality.

THIS IS THE NOVEL CONTRIBUTION:
    - Most XAI systems stop at visual explanations
    - We extend this by retrieving relevant medical knowledge
    - Creates a complete explanation pipeline with real citations
    - Semantic search ensures high-quality knowledge retrieval
"""

from .knowledge_base import MedicalKnowledgeBase
from .xai_to_text import XAITextConverter
from .explanation_generator import ExplanationGenerator

# Import PubMed retriever (optional - works offline without it)
try:
    from .pubmed_retriever import PubMedRetriever, PubMedArticle
    PUBMED_AVAILABLE = True
except ImportError:
    PUBMED_AVAILABLE = False
    PubMedRetriever = None
    PubMedArticle = None

# Import Semantic Search (optional - falls back to keyword matching)
try:
    from .semantic_search import (
        SemanticSearch,
        SemanticKnowledgeBase,
        SemanticPubMedSearch,
        SENTENCE_TRANSFORMERS_AVAILABLE
    )
    SEMANTIC_SEARCH_AVAILABLE = SENTENCE_TRANSFORMERS_AVAILABLE
except ImportError:
    SEMANTIC_SEARCH_AVAILABLE = False
    SemanticSearch = None
    SemanticKnowledgeBase = None
    SemanticPubMedSearch = None

__all__ = [
    # Core RAG components
    'MedicalKnowledgeBase',
    'XAITextConverter',
    'ExplanationGenerator',
    # PubMed integration
    'PubMedRetriever',
    'PubMedArticle',
    'PUBMED_AVAILABLE',
    # Semantic search
    'SemanticSearch',
    'SemanticKnowledgeBase',
    'SemanticPubMedSearch',
    'SEMANTIC_SEARCH_AVAILABLE',
]
