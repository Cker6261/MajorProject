"""Quick test of PubMed retriever."""
from src.rag.pubmed_retriever import PubMedRetriever

retriever = PubMedRetriever()

print("\n=== Testing PubMed Retriever ===")
results = retriever.get_cancer_knowledge('adenocarcinoma', max_articles=3)

print(f"\nFound {len(results)} articles:")
for r in results:
    title = r.get('title', 'No title')
    print(f"- {title[:70]}...")
    print(f"  Source: {r.get('source', 'Unknown')}")
    print()
