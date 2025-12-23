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

WHY RAG FOR THIS PROJECT?
    - Bridges the gap between visual XAI (Grad-CAM) and human understanding
    - Makes predictions more interpretable for non-technical users
    - Adds clinical context to model decisions

WHY SIMPLE KEYWORD MATCHING FOR REVIEW-1?
    - Demonstrates the concept without complex infrastructure
    - Easy to explain and defend in viva
    - Can be upgraded to semantic search later (future enhancement)

THIS IS THE NOVEL CONTRIBUTION:
    - Most XAI systems stop at visual explanations
    - We extend this by retrieving relevant medical knowledge
    - Creates a more complete explanation pipeline
"""

from .knowledge_base import MedicalKnowledgeBase
from .xai_to_text import XAITextConverter
from .explanation_generator import ExplanationGenerator

__all__ = [
    'MedicalKnowledgeBase',
    'XAITextConverter',
    'ExplanationGenerator'
]
