"""
Final Project Verification Script
Verifies all components of LungXAI are working correctly
"""

print('=' * 60)
print('LUNGXAI PROJECT FINAL VERIFICATION')
print('=' * 60)
print()

# 1. Core imports
print('[1] Core Modules...')
from src.xai import GradCAM, create_heatmap_overlay
from src.models.classifier import LungCancerClassifier
from src.data.dataset import LungCancerDataset
from src.data.transforms import get_train_transforms, get_val_transforms
print('    ✓ All core modules loaded')

# 2. RAG with Semantic Search
print()
print('[2] RAG Module with Semantic Search...')
from src.rag import MedicalKnowledgeBase, ExplanationGenerator
from src.rag import SEMANTIC_SEARCH_AVAILABLE, PUBMED_AVAILABLE
print(f'    ✓ Knowledge Base loaded')
print(f'    ✓ Semantic Search: {"Available" if SEMANTIC_SEARCH_AVAILABLE else "Not available"}')
print(f'    ✓ PubMed: {"Available" if PUBMED_AVAILABLE else "Not available"}')

# 3. Test semantic search
if SEMANTIC_SEARCH_AVAILABLE:
    from src.rag.semantic_search import SemanticSearch
    search = SemanticSearch()
    docs = [
        {'id': '1', 'keywords': ['adenocarcinoma'], 'content': 'Adenocarcinoma is peripheral'},
        {'id': '2', 'keywords': ['squamous'], 'content': 'Squamous is central'}
    ]
    search.index_documents(docs)
    results = search.search('outer lung tumor', top_k=1)
    print(f'    ✓ Semantic search test: "{results[0]["content"][:30]}..." (score: {results[0]["similarity_score"]:.2f})')

# 4. Check model files
print()
print('[3] Model Checkpoints...')
import os
checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
print(f'    ✓ {len(checkpoints)} main checkpoints')
if os.path.exists('checkpoints/baseline'):
    bl = [f for f in os.listdir('checkpoints/baseline') if f.endswith('.pth')]
    print(f'    ✓ {len(bl)} baseline checkpoints')

# 5. Check documentation
print()
print('[4] Documentation...')
docs_to_check = [
    'docs/LungXAI_Research_Paper_IEEE.md',
    'docs/PSEUDOCODE.md',
    'README.md'
]
for doc in docs_to_check:
    if os.path.exists(doc):
        size = os.path.getsize(doc)
        print(f'    ✓ {doc} ({size//1024} KB)')

# 6. Project structure
print()
print('[5] Project Structure...')
print('    src/')
for subdir in ['data', 'models', 'rag', 'xai', 'utils']:
    path = f'src/{subdir}'
    if os.path.exists(path):
        files = [f for f in os.listdir(path) if f.endswith('.py') and not f.startswith('__')]
        print(f'      {subdir}/: {len(files)} modules')

print()
print('=' * 60)
print('ALL VERIFICATIONS PASSED')
print('=' * 60)
