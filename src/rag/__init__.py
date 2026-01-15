# =============================================================================
# RAG MODULE (RETRIEVAL-AUGMENTED GENERATION)
# Converts XAI visual evidence to textual medical explanations
# =============================================================================
"""
RAG Module for Explainable Lung Cancer Classification.

This module contains:
    - MedicalKnowledgeBase: Local JSON-based medical knowledge store
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

THIS IS THE NOVEL CONTRIBUTION:
    - Most XAI systems stop at visual explanations
    - We extend this by retrieving relevant medical knowledge
    - Creates a complete explanation pipeline with real citations
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

__all__ = [
    'MedicalKnowledgeBase',
    'XAITextConverter',
    'ExplanationGenerator',
    'PubMedRetriever',
    'PubMedArticle',
    'PUBMED_AVAILABLE'
]
