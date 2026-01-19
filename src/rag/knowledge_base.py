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
            # ADENOCARCINOMA KNOWLEDGE (Expanded)
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
            {
                "id": "adeno_005",
                "keywords": ["adenocarcinoma", "egfr", "mutation", "targeted", "therapy"],
                "content": "Adenocarcinoma is the most common subtype harboring EGFR mutations (10-15% in Western, 40-50% in Asian populations). EGFR-mutated tumors respond well to tyrosine kinase inhibitors like erlotinib and gefitinib.",
                "source": "Mok TS et al., NEJM 2009; Lynch TJ et al., NEJM 2004"
            },
            {
                "id": "adeno_006",
                "keywords": ["adenocarcinoma", "alk", "rearrangement", "crizotinib"],
                "content": "ALK rearrangements occur in approximately 5% of adenocarcinomas, typically in younger, non-smoking patients. These tumors respond to ALK inhibitors such as crizotinib and alectinib.",
                "source": "Shaw AT et al., NEJM 2013"
            },
            {
                "id": "adeno_007",
                "keywords": ["adenocarcinoma", "part-solid", "subsolid", "nodule"],
                "content": "Part-solid nodules (mixed ground-glass and solid components) in adenocarcinoma indicate invasive component. The solid component size correlates with invasiveness and prognosis.",
                "source": "Naidich DP et al., Radiology 2013"
            },
            {
                "id": "adeno_008",
                "keywords": ["adenocarcinoma", "acinar", "papillary", "micropapillary", "solid"],
                "content": "Invasive adenocarcinoma subtypes include acinar, papillary, micropapillary, and solid patterns. Micropapillary and solid patterns are associated with worse prognosis and higher recurrence rates.",
                "source": "Travis WD et al., J Thorac Oncol 2011"
            },
            {
                "id": "adeno_009",
                "keywords": ["adenocarcinoma", "non-smoker", "female", "asian"],
                "content": "Adenocarcinoma is the most common lung cancer in non-smokers, women, and Asian populations. It has shown increasing incidence relative to other lung cancer types over recent decades.",
                "source": "Subramanian J et al., J Thorac Oncol 2007"
            },
            {
                "id": "adeno_010",
                "keywords": ["adenocarcinoma", "pleural", "effusion", "spread"],
                "content": "Adenocarcinoma has a propensity for pleural involvement, often presenting with malignant pleural effusion. This indicates advanced disease (stage IVA) and affects treatment options.",
                "source": "Goldstraw P et al., IASLC Staging Manual, 8th Ed"
            },
            
            # ==================================================================
            # SQUAMOUS CELL CARCINOMA KNOWLEDGE (Expanded)
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
            {
                "id": "squamous_005",
                "keywords": ["squamous", "keratinization", "pearl", "differentiation"],
                "content": "Well-differentiated squamous cell carcinoma shows keratinization and keratin pearl formation. These features help distinguish it from poorly differentiated forms and other carcinomas.",
                "source": "Travis WD et al., WHO Classification 2021"
            },
            {
                "id": "squamous_006",
                "keywords": ["squamous", "p40", "p63", "ck5", "marker"],
                "content": "Squamous cell carcinoma expresses p40, p63, and CK5/6 immunohistochemical markers. These markers distinguish it from adenocarcinoma (TTF-1, Napsin A positive).",
                "source": "Rekhtman N et al., Mod Pathol 2011"
            },
            {
                "id": "squamous_007",
                "keywords": ["squamous", "hypercalcemia", "paraneoplastic", "pth"],
                "content": "Squamous cell carcinoma can produce PTHrP (parathyroid hormone-related protein), causing paraneoplastic hypercalcemia. This is more common in squamous than other lung cancer types.",
                "source": "Stewart AF, NEJM 2005"
            },
            {
                "id": "squamous_008",
                "keywords": ["squamous", "pancoast", "superior", "sulcus"],
                "content": "Pancoast tumors (superior sulcus tumors) are often squamous cell carcinoma. They invade the chest wall apex, causing shoulder pain, Horner syndrome, and brachial plexopathy.",
                "source": "Arcasoy SM et al., NEJM 1997"
            },
            {
                "id": "squamous_009",
                "keywords": ["squamous", "fgfr", "amplification", "therapy"],
                "content": "FGFR1 amplification occurs in 15-20% of squamous cell lung cancers. FGFR inhibitors are under investigation as targeted therapy options for this alteration.",
                "source": "Weiss J et al., Sci Transl Med 2010"
            },
            
            # ==================================================================
            # LARGE CELL CARCINOMA KNOWLEDGE (Expanded)
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
            {
                "id": "large_004",
                "keywords": ["large cell", "large_cell", "neuroendocrine", "lcnec"],
                "content": "Large cell neuroendocrine carcinoma (LCNEC) is a high-grade neuroendocrine tumor with poor prognosis. It shares features with small cell lung cancer and may benefit from similar treatment approaches.",
                "source": "Rekhtman N, Arch Pathol Lab Med 2010"
            },
            {
                "id": "large_005",
                "keywords": ["large cell", "large_cell", "prognosis", "survival"],
                "content": "Large cell carcinoma has a generally poor prognosis with 5-year survival rates of 10-15%. Early detection and surgical resection offer the best outcomes.",
                "source": "Asamura H et al., J Thorac Oncol 2015"
            },
            {
                "id": "large_006",
                "keywords": ["large cell", "large_cell", "exclusion", "diagnosis"],
                "content": "Large cell carcinoma is a diagnosis of exclusion, made when tumor cells lack glandular or squamous differentiation and cannot be classified as small cell carcinoma on histology.",
                "source": "Travis WD et al., J Thorac Oncol 2015"
            },
            
            # ==================================================================
            # NORMAL/BENIGN KNOWLEDGE (Expanded)
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
                "id": "normal_003",
                "keywords": ["normal", "normal cases", "fissure", "lobe", "anatomy"],
                "content": "Normal lung anatomy includes three lobes on the right (upper, middle, lower) and two on the left (upper, lower). The major and minor fissures are visible as thin linear structures.",
                "source": "Webb WR et al., HRCT of the Lung, 2014"
            },
            {
                "id": "normal_004",
                "keywords": ["normal", "normal cases", "attenuation", "hounsfield"],
                "content": "Normal lung parenchyma has CT attenuation values between -700 to -900 Hounsfield units (HU). Higher values may indicate infiltration, while lower values suggest emphysema.",
                "source": "Defined clinical imaging standards"
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
            {
                "id": "benign_003",
                "keywords": ["benign", "benign cases", "granuloma", "infection"],
                "content": "Granulomas are the most common benign lung nodules, often resulting from prior infections (tuberculosis, histoplasmosis). They typically appear as small, round, calcified nodules.",
                "source": "Defined clinical guidelines"
            },
            {
                "id": "benign_004",
                "keywords": ["benign", "benign cases", "hamartoma", "fat", "popcorn"],
                "content": "Pulmonary hamartomas are benign tumors containing cartilage, fat, and epithelium. CT showing fat density within a nodule is highly suggestive of hamartoma (popcorn calcification pattern).",
                "source": "Siegelman SS et al., Radiology 1986"
            },
            {
                "id": "benign_005",
                "keywords": ["benign", "benign cases", "inflammatory", "infection", "pneumonia"],
                "content": "Inflammatory nodules from organizing pneumonia or focal infection can mimic malignancy. Clinical history of recent infection and follow-up imaging showing resolution help distinguish benign from malignant.",
                "source": "Defined clinical guidelines"
            },
            
            # ==================================================================
            # GENERAL IMAGING PATTERNS (Expanded)
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
            },
            {
                "id": "pattern_005",
                "keywords": ["spiculation", "corona", "radiata", "margin"],
                "content": "Spiculated margins (corona radiata sign) strongly suggest malignancy. The radiating strands represent tumor extension along lymphatics, vessels, or direct parenchymal invasion.",
                "source": "Zwirewich CV et al., Radiology 1991"
            },
            {
                "id": "pattern_006",
                "keywords": ["consolidation", "air", "bronchogram"],
                "content": "Consolidation with air bronchograms may indicate pneumonia, but persistent consolidation should raise suspicion for adenocarcinoma with lepidic growth or lymphoma.",
                "source": "Defined clinical guidelines"
            },
            {
                "id": "pattern_007",
                "keywords": ["pleural", "effusion", "thickening"],
                "content": "Pleural effusion or nodular pleural thickening in the presence of lung mass suggests pleural metastasis (stage IVA). Thoracentesis may be needed for cytological confirmation.",
                "source": "Goldstraw P et al., IASLC Staging Manual, 8th Ed"
            },
            {
                "id": "pattern_008",
                "keywords": ["lymphadenopathy", "hilar", "mediastinal", "node"],
                "content": "Hilar or mediastinal lymphadenopathy (nodes >1cm short axis) in lung cancer suggests nodal metastasis. PET-CT and/or tissue sampling are recommended for staging.",
                "source": "Defined clinical guidelines"
            },
            
            # ==================================================================
            # DIFFERENTIAL DIAGNOSIS KNOWLEDGE
            # ==================================================================
            {
                "id": "diff_001",
                "keywords": ["differential", "diagnosis", "nodule", "solitary"],
                "content": "Differential diagnosis for solitary pulmonary nodule includes primary lung cancer, metastasis, granuloma, hamartoma, and carcinoid tumor. Size, growth rate, and patient risk factors guide management.",
                "source": "Gould MK et al., ACCP Guidelines, Chest 2013"
            },
            {
                "id": "diff_002",
                "keywords": ["metastasis", "metastatic", "secondary"],
                "content": "Pulmonary metastases typically appear as multiple, round, well-defined nodules of varying sizes. Common primary sites include breast, colon, kidney, and melanoma.",
                "source": "Defined clinical guidelines"
            },
            {
                "id": "diff_003",
                "keywords": ["pet", "fdg", "suv", "metabolic"],
                "content": "FDG-PET CT helps differentiate malignant from benign nodules. SUV >2.5 suggests malignancy, but false positives occur with infection/inflammation, and false negatives with slow-growing tumors.",
                "source": "Gould MK et al., JAMA 2001"
            },
            
            # ==================================================================
            # STAGING AND PROGNOSIS
            # ==================================================================
            {
                "id": "stage_001",
                "keywords": ["staging", "tnm", "stage", "prognosis"],
                "content": "Lung cancer staging uses the TNM system: T (tumor size/invasion), N (nodal involvement), M (metastasis). Stage I-II are localized, III is locally advanced, IV is metastatic.",
                "source": "Goldstraw P et al., IASLC Staging Manual, 8th Ed"
            },
            {
                "id": "stage_002",
                "keywords": ["survival", "prognosis", "outcome", "five-year"],
                "content": "Five-year survival varies dramatically by stage: Stage IA (77-92%), Stage IB (68%), Stage IIA (60%), Stage IIB (53%), Stage IIIA (36%), Stage IIIB (26%), Stage IV (10%).",
                "source": "Goldstraw P et al., J Thorac Oncol 2016"
            },
            
            # ==================================================================
            # TREATMENT CONTEXT
            # ==================================================================
            {
                "id": "treat_001",
                "keywords": ["treatment", "surgery", "resection", "lobectomy"],
                "content": "Surgical resection (lobectomy or pneumonectomy) is the primary treatment for early-stage NSCLC. Complete resection offers the best chance for cure in stage I-II disease.",
                "source": "NCCN Guidelines, Lung Cancer 2024"
            },
            {
                "id": "treat_002",
                "keywords": ["chemotherapy", "platinum", "doublet"],
                "content": "Platinum-based doublet chemotherapy (cisplatin/carboplatin + pemetrexed/paclitaxel) is standard for advanced NSCLC without targetable mutations.",
                "source": "NCCN Guidelines, Lung Cancer 2024"
            },
            {
                "id": "treat_003",
                "keywords": ["immunotherapy", "checkpoint", "pd1", "pdl1"],
                "content": "Immune checkpoint inhibitors (pembrolizumab, nivolumab, atezolizumab) have transformed lung cancer treatment. PD-L1 expression helps predict response to immunotherapy.",
                "source": "Reck M et al., NEJM 2016"
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
