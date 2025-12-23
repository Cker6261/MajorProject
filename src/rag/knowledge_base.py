# =============================================================================
# MEDICAL KNOWLEDGE BASE
# Local knowledge store for RAG-based explanation generation
# =============================================================================
"""
Medical Knowledge Base for Lung Cancer Explanation Generation.

WHAT IS THIS?
    A structured repository of medical knowledge about lung cancer types,
    imaging characteristics, and diagnostic criteria.

WHY LOCAL KNOWLEDGE BASE (NOT LLM)?
    1. Controlled Content: All facts are verified and citable
    2. No Hallucination: Unlike LLMs, responses are deterministic
    3. Explainable: Easy to show faculty exactly what knowledge is used
    4. No API Dependency: Works offline, no cost
    5. Sufficient for Review-1: Demonstrates the concept

FUTURE ENHANCEMENT:
    - Use sentence-transformers for semantic retrieval
    - Add more medical literature entries
    - Connect to medical ontologies (SNOMED, RadLex)

STRUCTURE:
    Knowledge entries have:
    - keywords: Terms that trigger retrieval
    - content: Medical information
    - source: Citation for academic credibility
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict


class MedicalKnowledgeBase:
    """
    Local medical knowledge base for lung cancer explanations.
    
    This class:
        - Stores medical knowledge as keyword-indexed entries
        - Retrieves relevant facts based on query keywords
        - Returns structured information for explanation generation
    
    Example:
        >>> kb = MedicalKnowledgeBase()
        >>> results = kb.retrieve("adenocarcinoma peripheral opacity")
        >>> for entry in results:
        ...     print(entry['content'])
    """
    
    def __init__(self, knowledge_file: Optional[str] = None):
        """
        Initialize the knowledge base.
        
        Args:
            knowledge_file: Path to JSON file with knowledge entries.
                          If None, uses built-in default knowledge.
        """
        self.entries: List[Dict] = []
        self.keyword_index: Dict[str, List[int]] = defaultdict(list)
        
        if knowledge_file and Path(knowledge_file).exists():
            self._load_from_file(knowledge_file)
        else:
            self._load_default_knowledge()
        
        self._build_index()
        print(f"✓ Knowledge base loaded with {len(self.entries)} entries")
    
    def _load_from_file(self, filepath: str) -> None:
        """Load knowledge from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.entries = data.get('entries', [])
    
    def _load_default_knowledge(self) -> None:
        """
        Load default medical knowledge about lung cancer.
        
        This is curated from medical literature for academic use.
        Each entry includes a source for citation.
        """
        self.entries = [
            # ==================================================================
            # ADENOCARCINOMA KNOWLEDGE
            # ==================================================================
            {
                "id": "adeno_001",
                "keywords": ["adenocarcinoma", "peripheral", "outer"],
                "content": "Adenocarcinoma typically presents in the peripheral regions of the lung, often in the outer third of the lung parenchyma. This peripheral location is a key distinguishing feature from other lung cancer types.",
                "source": "Travis WD et al., WHO Classification of Tumours of the Lung, 2021"
            },
            {
                "id": "adeno_002", 
                "keywords": ["adenocarcinoma", "ground glass", "opacity", "ggo"],
                "content": "Ground-glass opacity (GGO) on CT imaging is frequently associated with adenocarcinoma, particularly the lepidic subtype. GGO appears as a hazy increase in lung attenuation without obscuration of underlying vessels.",
                "source": "Hansell DM et al., Fleischner Society Glossary, Radiology 2008"
            },
            {
                "id": "adeno_003",
                "keywords": ["adenocarcinoma", "nodule", "spiculated", "irregular"],
                "content": "Adenocarcinoma often presents as a spiculated nodule with irregular margins. Spiculation refers to radiating strands extending from the nodule margin, which suggests invasive growth pattern.",
                "source": "MacMahon H et al., Lung Nodule Guidelines, Radiology 2017"
            },
            {
                "id": "adeno_004",
                "keywords": ["adenocarcinoma", "bronchoalveolar", "lepidic"],
                "content": "The lepidic growth pattern (formerly bronchoalveolar carcinoma) shows tumor cells growing along intact alveolar walls. This pattern often correlates with ground-glass appearance on CT.",
                "source": "Travis WD et al., IASLC/ATS/ERS Classification, J Thorac Oncol 2011"
            },
            
            # ==================================================================
            # SQUAMOUS CELL CARCINOMA KNOWLEDGE
            # ==================================================================
            {
                "id": "squamous_001",
                "keywords": ["squamous", "central", "hilum", "hilar"],
                "content": "Squamous cell carcinoma typically arises in the central airways, near the hilum. This central location reflects its origin from the bronchial epithelium of major airways.",
                "source": "Travis WD et al., WHO Classification of Tumours of the Lung, 2021"
            },
            {
                "id": "squamous_002",
                "keywords": ["squamous", "cavitation", "cavity", "necrosis"],
                "content": "Squamous cell carcinoma has the highest propensity for cavitation among lung cancers. Cavitation occurs due to central tumor necrosis and appears as a lucent area within the mass.",
                "source": "Chaudhuri MR, Br J Dis Chest 1973"
            },
            {
                "id": "squamous_003",
                "keywords": ["squamous", "bronchus", "airway", "obstruction"],
                "content": "Squamous cell carcinoma frequently causes bronchial obstruction leading to post-obstructive atelectasis or pneumonia. The tumor often grows endobronchially into the airway lumen.",
                "source": "Colby TV et al., Tumors of the Lower Respiratory Tract, AFIP 1995"
            },
            {
                "id": "squamous_004",
                "keywords": ["squamous", "smoking", "tobacco"],
                "content": "Squamous cell carcinoma has the strongest association with tobacco smoking among lung cancer types. It typically develops through a sequence of squamous metaplasia and dysplasia.",
                "source": "Kenfield SA et al., JAMA 2008"
            },
            
            # ==================================================================
            # LARGE CELL CARCINOMA KNOWLEDGE
            # ==================================================================
            {
                "id": "large_001",
                "keywords": ["large cell", "large_cell", "peripheral", "mass"],
                "content": "Large cell carcinoma typically presents as a large peripheral mass. It is characterized by rapid growth and often reaches considerable size before diagnosis.",
                "source": "Travis WD et al., WHO Classification of Tumours of the Lung, 2021"
            },
            {
                "id": "large_002",
                "keywords": ["large cell", "large_cell", "undifferentiated", "aggressive"],
                "content": "Large cell carcinoma is an undifferentiated non-small cell lung cancer that lacks the specific features of adenocarcinoma or squamous cell carcinoma. It tends to be aggressive with early metastasis.",
                "source": "Rossi G et al., Lung Cancer 2003"
            },
            {
                "id": "large_003",
                "keywords": ["large cell", "large_cell", "necrosis", "heterogeneous"],
                "content": "Large cell carcinoma often shows heterogeneous enhancement on CT with areas of necrosis. The tumor margins may be well-defined or irregular.",
                "source": "Truong MT et al., Radiographics 2014"
            },
            
            # ==================================================================
            # NORMAL/BENIGN KNOWLEDGE
            # ==================================================================
            {
                "id": "normal_001",
                "keywords": ["normal", "normal cases", "healthy", "clear", "unremarkable"],
                "content": "Normal lung parenchyma appears as low-attenuation tissue with visible pulmonary vessels. The lung fields should be clear without focal opacities, masses, or nodules.",
                "source": "Webb WR et al., High-Resolution CT of the Lung, 2014"
            },
            {
                "id": "normal_002",
                "keywords": ["normal", "normal cases", "vessel", "vascular", "bronchovascular"],
                "content": "In normal CT imaging, bronchovascular structures are visible extending from the hilum. The pulmonary arteries accompany bronchi, while veins course independently toward the left atrium.",
                "source": "Naidich DP et al., Computed Tomography of the Thorax, 2006"
            },
            {
                "id": "benign_001",
                "keywords": ["benign", "benign cases", "non-malignant", "non-cancerous"],
                "content": "Benign lung lesions include granulomas, hamartomas, and inflammatory nodules. These lesions typically have smooth, well-defined margins and stable size over time on serial imaging.",
                "source": "Defined clinical guidelines"
            },
            {
                "id": "benign_002",
                "keywords": ["benign", "benign cases", "calcification", "stable"],
                "content": "Benign nodules often demonstrate characteristic calcification patterns such as central, laminated, or popcorn calcification. Stability over 2 years is a strong indicator of benign etiology.",
                "source": "MacMahon H et al., Lung Nodule Guidelines, Radiology 2017"
            },
            
            # ==================================================================
            # GENERAL IMAGING PATTERNS
            # ==================================================================
            {
                "id": "pattern_001",
                "keywords": ["upper", "lobe", "apical"],
                "content": "Upper lobe predominance in lung lesions may suggest primary lung cancer or reactivation of tuberculosis. Careful evaluation of morphology is essential for differentiation.",
                "source": "Defined clinical guidelines"
            },
            {
                "id": "pattern_002",
                "keywords": ["lower", "lobe", "basal"],
                "content": "Lower lobe lesions require consideration of aspiration-related pathology in addition to primary malignancy. The posterior basal segments are common sites for aspiration.",
                "source": "Defined clinical guidelines"
            },
            {
                "id": "pattern_003",
                "keywords": ["central", "mediastinal", "lymph"],
                "content": "Central lesions with mediastinal involvement may indicate locally advanced disease. Lymph node enlargement in the mediastinum suggests possible nodal metastasis.",
                "source": "Mountain CF, Chest 1997"
            },
            {
                "id": "pattern_004",
                "keywords": ["intensity", "bright", "high", "activation"],
                "content": "High-intensity regions in the model's attention map indicate areas the neural network considers most relevant for classification. These regions should correspond to pathologically significant features.",
                "source": "Selvaraju RR et al., Grad-CAM, ICCV 2017"
            }
        ]
    
    def _build_index(self) -> None:
        """Build keyword index for fast retrieval."""
        self.keyword_index.clear()
        
        for idx, entry in enumerate(self.entries):
            for keyword in entry.get('keywords', []):
                # Index by lowercase keyword
                self.keyword_index[keyword.lower()].append(idx)
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 3,
        min_matches: int = 1
    ) -> List[Dict]:
        """
        Retrieve relevant knowledge entries based on query.
        
        Uses keyword matching (simple but effective for Review-1).
        
        Args:
            query: Query string containing keywords
            top_k: Maximum number of entries to return
            min_matches: Minimum keyword matches required
        
        Returns:
            List of relevant knowledge entries, sorted by relevance
        """
        query_words = query.lower().split()
        
        # Count matches for each entry
        entry_scores = defaultdict(int)
        
        for word in query_words:
            # Check exact matches
            if word in self.keyword_index:
                for idx in self.keyword_index[word]:
                    entry_scores[idx] += 2  # Exact match bonus
            
            # Check partial matches
            for keyword, indices in self.keyword_index.items():
                if word in keyword or keyword in word:
                    for idx in indices:
                        entry_scores[idx] += 1
        
        # Filter and sort by score
        relevant = [
            (idx, score) for idx, score in entry_scores.items()
            if score >= min_matches
        ]
        relevant.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k entries
        results = []
        for idx, score in relevant[:top_k]:
            entry = self.entries[idx].copy()
            entry['relevance_score'] = score
            results.append(entry)
        
        return results
    
    def get_class_knowledge(self, class_name: str) -> List[Dict]:
        """
        Get all knowledge entries for a specific cancer class.
        
        Args:
            class_name: Name of the cancer class
        
        Returns:
            List of knowledge entries for that class
        """
        # Map class names to knowledge base keywords
        # Handles both original names and folder names with spaces
        class_keywords = {
            'adenocarcinoma': ['adenocarcinoma'],
            'squamous_cell_carcinoma': ['squamous'],
            'squamous cell carcinoma': ['squamous'],
            'large_cell_carcinoma': ['large cell', 'large_cell'],
            'large cell carcinoma': ['large cell', 'large_cell'],
            'normal': ['normal', 'healthy'],
            'normal cases': ['normal', 'healthy'],
            'benign': ['benign', 'normal'],
            'benign cases': ['benign', 'normal', 'healthy']
        }
        
        keywords = class_keywords.get(class_name.lower(), [class_name.lower()])
        
        results = []
        for keyword in keywords:
            results.extend(self.retrieve(keyword, top_k=5))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for entry in results:
            if entry['id'] not in seen:
                seen.add(entry['id'])
                unique_results.append(entry)
        
        return unique_results
    
    def add_entry(self, entry: Dict) -> None:
        """
        Add a new knowledge entry.
        
        Args:
            entry: Dictionary with 'id', 'keywords', 'content', 'source'
        """
        self.entries.append(entry)
        idx = len(self.entries) - 1
        
        for keyword in entry.get('keywords', []):
            self.keyword_index[keyword.lower()].append(idx)
    
    def save_to_file(self, filepath: str) -> None:
        """Save knowledge base to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({'entries': self.entries}, f, indent=2, ensure_ascii=False)
        print(f"✓ Knowledge base saved to {filepath}")
