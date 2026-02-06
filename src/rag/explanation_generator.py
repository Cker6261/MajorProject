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
    │ CLINICAL KNOWLEDGE BASE                                     │
    │ ----------------------------------------                    │
    │ • Adenocarcinoma typically presents in the peripheral       │
    │   regions of the lung. [1]                                  │
    │                                                             │
    │ RECENT RESEARCH (PubMed)                                    │
    │ ----------------------------------------                    │
    │ • Ground-glass opacity is frequently associated with        │
    │   this cancer type on CT imaging. [2]                       │
    ├────────────────────────────────────────────────────────────┤
    │ REFERENCES:                                                 │
    │   [1] Travis WD et al., WHO Classification 2021             │
    │   [2] Lung adenocarcinoma CT features. PMID: 28463456       │
    └────────────────────────────────────────────────────────────┘

WHY THIS APPROACH?
    1. Transparent: User can see exactly what evidence is used
    2. Citable: Each fact has a source reference with citation numbers
    3. Educational: Explains both model behavior and medical context
    4. Trustworthy: Separates AI observation from medical knowledge
    5. Organized: Knowledge Base and PubMed sources clearly separated
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
                    max_articles=4  # Fetch more to account for duplicates after deduplication
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
        """Format the medical context section and extract sources with citation markers."""
        if not knowledge_entries and not pubmed_entries:
            return "No specific medical context available for this pattern.", []
        
        context_parts = []
        sources = []  # Will store tuples of (citation_num, source_text, year/date for sorting, is_pubmed)
        citation_num = 1
        
        # First, process PubMed entries (these should come first as they are more recent)
        pubmed_sources = []
        seen_pmids = set()  # Track seen PMIDs to avoid duplicates
        seen_content = set()  # Track seen content to avoid duplicate text
        if pubmed_entries:
            # Sort PubMed entries by date (latest first)
            sorted_pubmed = sorted(pubmed_entries, 
                                   key=lambda x: x.get('pub_date', '2000')[:4] if x.get('pub_date') else '2000', 
                                   reverse=True)
            for entry in sorted_pubmed:
                pmid = entry.get('pmid', 'Unknown')
                content = entry.get('content', '')
                
                # Skip duplicates by PMID
                if pmid in seen_pmids:
                    continue
                
                # Skip duplicates by content (first 100 chars)
                content_key = content[:100].lower() if content else ''
                if content_key and content_key in seen_content:
                    continue
                
                seen_pmids.add(pmid)
                if content_key:
                    seen_content.add(content_key)
                
                title = entry.get('title', 'Untitled')
                pub_date = entry.get('pub_date', 'Unknown')
                journal = entry.get('journal', '')
                
                # Clean and truncate content
                if content and len(content) > 350:
                    content = content[:350] + "..."
                
                source_text = f"{title}. {journal}, {pub_date}. PMID: {pmid}"
                pubmed_sources.append({
                    'content': content,
                    'source': source_text,
                    'pub_date': pub_date
                })
        
        # Collect all sources with dates for sorting (latest first in references)
        all_sources_with_dates = []  # (citation_num, source_text, date_for_sorting)
        
        # Add local knowledge base entries first
        if knowledge_entries:
            context_parts.append("【 CLINICAL KNOWLEDGE BASE 】")
            context_parts.append("")
            for i, entry in enumerate(knowledge_entries):
                content = entry['content']
                source = entry.get('source', 'Medical Knowledge Base')
                # Add citation marker to the text
                context_parts.append(f"• {content} [{citation_num}]")
                if i < len(knowledge_entries) - 1:
                    context_parts.append("")  # Space between entries
                # Extract year from source for sorting
                year = "2020"
                for y in ['2021', '2020', '2019', '2018', '2017', '2015', '2013', '2008', '2003', '1973']:
                    if y in source:
                        year = y
                        break
                all_sources_with_dates.append((citation_num, source, year))
                citation_num += 1
        
        # Add PubMed entries (already sorted by date, limit to 2)
        if pubmed_sources:
            if knowledge_entries:
                context_parts.append("")
                context_parts.append("")
                context_parts.append("─" * 40)
                context_parts.append("")
            context_parts.append("【 RECENT RESEARCH (PubMed) 】")
            context_parts.append("")
            for i, entry in enumerate(pubmed_sources[:2]):
                content = entry['content']
                if content:
                    context_parts.append(f"• {content} [{citation_num}]")
                    if i < min(2, len(pubmed_sources)) - 1:
                        context_parts.append("")
                    # Extract year for sorting
                    pub_date = entry.get('pub_date', '2020')
                    year = pub_date[:4] if pub_date and len(pub_date) >= 4 else '2020'
                    all_sources_with_dates.append((citation_num, entry['source'], year))
                    citation_num += 1
        
        medical_context = '\n'.join(context_parts)
        
        # Sort all sources by year (latest first) and format
        all_sources_with_dates.sort(key=lambda x: x[2], reverse=True)
        sources = [f"[{num}] {src}" for num, src, _ in all_sources_with_dates]
        
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
        lines.append("MEDICAL CONTEXT:")
        lines.append("-" * 60)
        lines.append(medical_context)
        
        # References (separate section with citations)
        if sources:
            lines.append("")
            lines.append("-" * 60)
            lines.append("REFERENCES:")
            lines.append("-" * 60)
            for source in sources:
                lines.append(f"  {source}")
        
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
