# =============================================================================
# EXPLANATION GENERATOR
# Combines XAI analysis with retrieved knowledge to generate explanations
# =============================================================================
"""
Explanation Generator: The Final Step in the RAG Pipeline.

WHAT DOES THIS DO?
    Combines three sources of information:
    1. Model prediction (class + confidence)
    2. XAI analysis (where the model is looking)
    3. Retrieved knowledge (medical context)
    
    → Generates a coherent, evidence-based explanation

OUTPUT FORMAT:
    ┌────────────────────────────────────────────────────────────┐
    │ PREDICTION: Adenocarcinoma (92.3%)                         │
    ├────────────────────────────────────────────────────────────┤
    │ VISUAL EVIDENCE:                                            │
    │ The model focused on the peripheral upper region with       │
    │ high attention intensity, concentrated in specific areas.   │
    ├────────────────────────────────────────────────────────────┤
    │ MEDICAL CONTEXT:                                            │
    │ Adenocarcinoma typically presents in the peripheral regions │
    │ of the lung. Ground-glass opacity is frequently associated  │
    │ with this cancer type.                                      │
    ├────────────────────────────────────────────────────────────┤
    │ Sources: Travis WD et al., WHO Classification 2021          │
    └────────────────────────────────────────────────────────────┘

WHY THIS APPROACH?
    1. Transparent: User can see exactly what evidence is used
    2. Citable: Each fact has a source reference
    3. Educational: Explains both model behavior and medical context
    4. Trustworthy: Separates AI observation from medical knowledge
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .knowledge_base import MedicalKnowledgeBase
from .xai_to_text import XAITextConverter


@dataclass
class Explanation:
    """
    Structured explanation output.
    
    Attributes:
        predicted_class: The predicted cancer type
        confidence: Prediction confidence (0-1)
        visual_evidence: Description of what the model focused on
        medical_context: Retrieved medical knowledge
        sources: List of cited sources
        full_explanation: Complete formatted explanation
        keywords_used: Keywords used for retrieval
    """
    predicted_class: str
    confidence: float
    visual_evidence: str
    medical_context: str
    sources: List[str]
    full_explanation: str
    keywords_used: List[str]
    
    def __str__(self):
        return self.full_explanation
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'visual_evidence': self.visual_evidence,
            'medical_context': self.medical_context,
            'sources': self.sources,
            'full_explanation': self.full_explanation,
            'keywords_used': self.keywords_used
        }


class ExplanationGenerator:
    """
    Generates evidence-based explanations by combining XAI and RAG.
    
    This is the main class that brings together:
        - XAITextConverter: Converts heatmaps to text
        - MedicalKnowledgeBase: Retrieves relevant medical facts
        - PubMedRetriever: Fetches real medical literature (optional)
    
    Example:
        >>> generator = ExplanationGenerator()
        >>> explanation = generator.generate(
        ...     heatmap=gradcam_output,
        ...     predicted_class="adenocarcinoma",
        ...     confidence=0.92
        ... )
        >>> print(explanation)
    """
    
    def __init__(
        self, 
        knowledge_file: Optional[str] = None, 
        use_pubmed: bool = True,
        use_semantic_search: bool = True
    ):
        """
        Initialize the explanation generator.
        
        Args:
            knowledge_file: Optional path to custom knowledge base JSON
            use_pubmed: Whether to use PubMed for additional knowledge retrieval
            use_semantic_search: Whether to use semantic (embedding-based) search
        """
        self.xai_converter = XAITextConverter()
        self.knowledge_base = MedicalKnowledgeBase(knowledge_file)
        
        # Initialize semantic search if available and requested
        self.use_semantic_search = use_semantic_search
        self.semantic_kb = None
        if use_semantic_search:
            try:
                from .semantic_search import SemanticKnowledgeBase, SENTENCE_TRANSFORMERS_AVAILABLE
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.semantic_kb = SemanticKnowledgeBase(use_gpu=True)
                    self.semantic_kb.load_from_knowledge_base(self.knowledge_base)
                    print("✓ Semantic search enabled for knowledge retrieval")
                else:
                    self.use_semantic_search = False
            except Exception as e:
                print(f"⚠ Semantic search disabled: {e}")
                self.use_semantic_search = False
        
        # Initialize PubMed retriever if enabled
        self.use_pubmed = use_pubmed
        self.pubmed_retriever = None
        self.semantic_pubmed = None
        if use_pubmed:
            try:
                from .pubmed_retriever import PubMedRetriever
                self.pubmed_retriever = PubMedRetriever(cache_dir="./pubmed_cache")
                
                # Also initialize semantic PubMed search if available
                if use_semantic_search:
                    try:
                        from .semantic_search import SemanticPubMedSearch
                        self.semantic_pubmed = SemanticPubMedSearch(use_gpu=True)
                    except Exception:
                        pass
            except Exception as e:
                print(f"⚠ PubMed integration disabled: {e}")
                self.use_pubmed = False
        
        status = []
        if self.use_semantic_search:
            status.append("semantic search")
        if self.use_pubmed:
            status.append("PubMed")
        status_str = f" (with {', '.join(status)})" if status else ""
        print(f"✓ Explanation Generator initialized{status_str}")
    
    def generate(
        self,
        heatmap: np.ndarray,
        predicted_class: str,
        confidence: float,
        top_k_knowledge: int = 2,
        include_pubmed: bool = True
    ) -> Explanation:
        """
        Generate a complete explanation for a prediction.
        
        Args:
            heatmap: Grad-CAM heatmap [H, W]
            predicted_class: Name of predicted class
            confidence: Prediction confidence (0-1)
            top_k_knowledge: Number of knowledge entries to retrieve
            include_pubmed: Whether to include PubMed articles in the explanation
        
        Returns:
            Explanation object with all components
        """
        # Step 1: Convert XAI to text
        xai_result = self.xai_converter.convert(
            heatmap=heatmap,
            predicted_class=predicted_class,
            confidence=confidence
        )
        
        # Step 2: Retrieve relevant knowledge
        # Use semantic search if available, otherwise fall back to keyword matching
        query = ' '.join(xai_result['keywords'])
        
        if self.use_semantic_search and self.semantic_kb is not None:
            # Semantic search - understands meaning, not just keywords
            knowledge_entries = self.semantic_kb.hybrid_search(
                query=query,
                class_name=predicted_class,
                top_k=top_k_knowledge
            )
        else:
            # Fallback to keyword matching
            knowledge_entries = self.knowledge_base.retrieve(query, top_k=top_k_knowledge)
        
        # If no matches, get class-specific knowledge
        if not knowledge_entries:
            knowledge_entries = self.knowledge_base.get_class_knowledge(predicted_class)[:top_k_knowledge]
        
        # Step 2b: Optionally add PubMed knowledge
        pubmed_entries = []
        if include_pubmed and self.use_pubmed and self.pubmed_retriever:
            try:
                pubmed_entries = self.pubmed_retriever.get_cancer_knowledge(
                    predicted_class, 
                    max_articles=2
                )
            except Exception as e:
                print(f"⚠ PubMed retrieval failed: {e}")
        
        # Step 3: Format visual evidence
        visual_evidence = self._format_visual_evidence(xai_result)
        
        # Step 4: Format medical context (combine local + PubMed)
        medical_context, sources = self._format_medical_context(knowledge_entries, pubmed_entries)
        
        # Step 5: Generate full explanation
        full_explanation = self._format_full_explanation(
            predicted_class=predicted_class,
            confidence=confidence,
            visual_evidence=visual_evidence,
            medical_context=medical_context,
            sources=sources
        )
        
        return Explanation(
            predicted_class=predicted_class,
            confidence=confidence,
            visual_evidence=visual_evidence,
            medical_context=medical_context,
            sources=sources,
            full_explanation=full_explanation,
            keywords_used=xai_result['keywords']
        )
    
    def _format_visual_evidence(self, xai_result: Dict) -> str:
        """Format the visual evidence section."""
        parts = []
        
        # Add spatial description
        parts.append(xai_result['spatial_description'])
        
        # Add intensity description
        parts.append(xai_result['intensity_description'])
        
        # Add focus region details if available
        focus_regions = xai_result.get('focus_regions', [])
        if focus_regions:
            region = focus_regions[0]
            parts.append(
                f"Primary attention focus is in the {region['location']} area, "
                f"covering approximately {region['size_ratio']*100:.1f}% of the image."
            )
        
        return ' '.join(parts)
    
    def _format_medical_context(self, knowledge_entries: List[Dict], pubmed_entries: List[Dict] = None) -> tuple:
        """Format the medical context section and extract sources."""
        if not knowledge_entries and not pubmed_entries:
            return "No specific medical context available for this pattern.", []
        
        context_parts = []
        sources = []
        
        # Add local knowledge base entries
        if knowledge_entries:
            context_parts.append("**Clinical Knowledge:**")
            for entry in knowledge_entries:
                context_parts.append(f"• {entry['content']}")
                source = entry.get('source', 'Unknown source')
                if source not in sources:
                    sources.append(source)
        
        # Add PubMed entries if available
        if pubmed_entries:
            context_parts.append("\n**Recent Research (PubMed):**")
            for entry in pubmed_entries:
                title = entry.get('title', 'Untitled')
                content = entry.get('content', '')[:300]
                if len(entry.get('content', '')) > 300:
                    content += "..."
                context_parts.append(f"• \"{title}\": {content}")
                source = entry.get('source', f"PMID: {entry.get('pmid', 'Unknown')}")
                if source not in sources:
                    sources.append(f"[PubMed] {source}")
        
        medical_context = '\n'.join(context_parts)
        
        return medical_context, sources
    
    def _format_full_explanation(
        self,
        predicted_class: str,
        confidence: float,
        visual_evidence: str,
        medical_context: str,
        sources: List[str]
    ) -> str:
        """Format the complete explanation text."""
        # Format class name nicely
        class_display = predicted_class.replace('_', ' ').title()
        
        # Build the explanation
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("EXPLAINABLE AI ANALYSIS REPORT")
        lines.append("=" * 60)
        
        # Prediction
        lines.append("")
        lines.append(f"PREDICTION: {class_display}")
        lines.append(f"CONFIDENCE: {confidence * 100:.1f}%")
        
        # Visual Evidence
        lines.append("")
        lines.append("-" * 60)
        lines.append("VISUAL EVIDENCE (What the model focused on):")
        lines.append("-" * 60)
        lines.append(visual_evidence)
        
        # Medical Context
        lines.append("")
        lines.append("-" * 60)
        lines.append("MEDICAL CONTEXT (Retrieved knowledge):")
        lines.append("-" * 60)
        lines.append(medical_context)
        
        # Sources
        if sources:
            lines.append("")
            lines.append("-" * 60)
            lines.append("SOURCES:")
            lines.append("-" * 60)
            for i, source in enumerate(sources, 1):
                lines.append(f"  [{i}] {source}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return '\n'.join(lines)
    
    def generate_short_explanation(
        self,
        heatmap: np.ndarray,
        predicted_class: str,
        confidence: float
    ) -> str:
        """
        Generate a short, concise explanation (for display in UI).
        
        Args:
            heatmap: Grad-CAM heatmap
            predicted_class: Predicted class name
            confidence: Confidence score
        
        Returns:
            Short explanation string
        """
        # Get XAI analysis
        xai_result = self.xai_converter.convert(heatmap, predicted_class, confidence)
        
        # Get one knowledge entry
        knowledge = self.knowledge_base.get_class_knowledge(predicted_class)
        
        # Format class name
        class_display = predicted_class.replace('_', ' ').title()
        
        # Build short explanation
        spatial = xai_result['spatial_info']
        pvc = spatial.get('peripheral_vs_central', {})
        
        location = "peripheral" if pvc.get('is_peripheral') else "central"
        
        explanation = f"Prediction: {class_display} ({confidence*100:.1f}%)\n\n"
        explanation += f"The model identified features in the {location} lung region. "
        
        if knowledge:
            # Add one key medical fact
            explanation += knowledge[0]['content'][:200]
            if len(knowledge[0]['content']) > 200:
                explanation += "..."
        
        return explanation


def create_explanation_pipeline():
    """
    Factory function to create a ready-to-use explanation pipeline.
    
    Returns:
        ExplanationGenerator instance
    """
    return ExplanationGenerator()
