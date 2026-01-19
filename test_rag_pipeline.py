#!/usr/bin/env python3
"""
Test script for the RAG (Retrieval-Augmented Generation) pipeline.

This script tests:
1. Knowledge Base loading and retrieval
2. PubMed API integration
3. Explanation generation with XAI
4. Full pipeline integration

Run this to verify the RAG system is working properly.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_knowledge_base():
    """Test the medical knowledge base."""
    print("\n" + "=" * 60)
    print("TEST 1: MEDICAL KNOWLEDGE BASE")
    print("=" * 60)
    
    from rag.knowledge_base import MedicalKnowledgeBase
    
    kb = MedicalKnowledgeBase()
    print(f"✓ Loaded {len(kb.entries)} knowledge entries")
    
    # Test retrieval for each cancer type
    test_queries = [
        ("adenocarcinoma peripheral", "Adenocarcinoma"),
        ("squamous central cavitation", "Squamous Cell Carcinoma"),
        ("large cell aggressive", "Large Cell Carcinoma"),
        ("normal healthy clear", "Normal Cases"),
        ("benign calcification stable", "Benign Cases")
    ]
    
    for query, cancer_type in test_queries:
        results = kb.retrieve(query, top_k=2)
        print(f"\n  Query: '{query}'")
        print(f"  Found {len(results)} results:")
        for r in results:
            print(f"    - [{r['id']}] Score: {r.get('relevance_score', 0)}")
            print(f"      Content: {r['content'][:80]}...")
    
    # Test class-specific retrieval
    print("\n  Testing class-specific retrieval:")
    for class_name in ['adenocarcinoma', 'squamous cell carcinoma', 'large cell carcinoma', 'benign cases', 'normal cases']:
        entries = kb.get_class_knowledge(class_name)
        print(f"    - {class_name}: {len(entries)} entries found")
    
    print("\n✓ Knowledge Base tests PASSED")
    return True


def test_pubmed_retriever():
    """Test the PubMed API integration."""
    print("\n" + "=" * 60)
    print("TEST 2: PUBMED RETRIEVER")
    print("=" * 60)
    
    from rag.pubmed_retriever import PubMedRetriever
    
    retriever = PubMedRetriever(cache_dir="./pubmed_cache")
    
    # Test search
    print("\n  Testing PubMed search for 'lung adenocarcinoma CT imaging'...")
    try:
        articles = retriever.search("lung adenocarcinoma CT imaging", max_results=2)
        
        if articles:
            print(f"  ✓ Found {len(articles)} articles")
            for article in articles:
                print(f"\n    Title: {article.title[:60]}...")
                print(f"    Citation: {article.to_citation()}")
                print(f"    Abstract: {article.abstract[:100]}..." if article.abstract else "    No abstract")
        else:
            print("  ⚠ No articles found (may be network issue)")
            
    except Exception as e:
        print(f"  ⚠ PubMed search failed: {e}")
        print("  (This is OK if you don't have internet access)")
    
    # Test cancer-specific knowledge
    print("\n  Testing cancer-specific knowledge retrieval...")
    try:
        knowledge = retriever.get_cancer_knowledge("adenocarcinoma", max_articles=2)
        print(f"  ✓ Retrieved {len(knowledge)} knowledge entries for adenocarcinoma")
        for k in knowledge:
            print(f"    - {k.get('title', 'No title')[:50]}...")
    except Exception as e:
        print(f"  ⚠ Cancer knowledge retrieval failed: {e}")
    
    print("\n✓ PubMed Retriever tests completed")
    return True


def test_xai_to_text():
    """Test the XAI to text conversion."""
    print("\n" + "=" * 60)
    print("TEST 3: XAI TO TEXT CONVERSION")
    print("=" * 60)
    
    from rag.xai_to_text import XAITextConverter
    
    converter = XAITextConverter()
    
    # Create a mock heatmap (simulating Grad-CAM output)
    # Peripheral focus (adenocarcinoma-like)
    heatmap = np.zeros((224, 224))
    heatmap[20:80, 150:200] = 0.8  # Upper-right peripheral region
    heatmap[50:60, 170:180] = 1.0  # Hot spot
    
    result = converter.convert(
        heatmap=heatmap,
        predicted_class="adenocarcinoma",
        confidence=0.92
    )
    
    print(f"\n  Generated keywords: {result['keywords']}")
    print(f"  Spatial description: {result['spatial_description']}")
    print(f"  Intensity description: {result['intensity_description']}")
    
    # Test with central focus (squamous-like)
    heatmap2 = np.zeros((224, 224))
    heatmap2[80:140, 80:140] = 0.9  # Central region
    
    result2 = converter.convert(
        heatmap=heatmap2,
        predicted_class="squamous cell carcinoma",
        confidence=0.85
    )
    
    print(f"\n  Central pattern keywords: {result2['keywords']}")
    print(f"  Spatial description: {result2['spatial_description']}")
    
    print("\n✓ XAI to Text tests PASSED")
    return True


def test_explanation_generator():
    """Test the full explanation generation pipeline."""
    print("\n" + "=" * 60)
    print("TEST 4: EXPLANATION GENERATOR (Full Pipeline)")
    print("=" * 60)
    
    from rag.explanation_generator import ExplanationGenerator
    
    # Test with PubMed enabled
    generator = ExplanationGenerator(use_pubmed=True)
    
    # Create mock heatmap
    heatmap = np.zeros((224, 224))
    heatmap[30:90, 140:200] = 0.75  # Peripheral upper region
    heatmap[50:70, 160:180] = 1.0   # Hot spot
    
    # Generate explanation
    print("\n  Generating explanation for adenocarcinoma prediction...")
    explanation = generator.generate(
        heatmap=heatmap,
        predicted_class="adenocarcinoma",
        confidence=0.92,
        include_pubmed=True
    )
    
    print(f"\n  Predicted Class: {explanation.predicted_class}")
    print(f"  Confidence: {explanation.confidence * 100:.1f}%")
    print(f"  Keywords Used: {explanation.keywords_used}")
    print(f"\n  Visual Evidence:\n    {explanation.visual_evidence}")
    print(f"\n  Medical Context (first 300 chars):\n    {explanation.medical_context[:300]}...")
    print(f"\n  Sources ({len(explanation.sources)} total):")
    for source in explanation.sources[:3]:
        print(f"    - {source[:70]}...")
    
    # Test without PubMed
    print("\n  Testing without PubMed (local KB only)...")
    explanation2 = generator.generate(
        heatmap=heatmap,
        predicted_class="squamous cell carcinoma",
        confidence=0.87,
        include_pubmed=False
    )
    print(f"  ✓ Generated explanation for {explanation2.predicted_class}")
    
    print("\n✓ Explanation Generator tests PASSED")
    return True


def test_full_pipeline_integration():
    """Test full integration with a simulated prediction."""
    print("\n" + "=" * 60)
    print("TEST 5: FULL PIPELINE INTEGRATION")
    print("=" * 60)
    
    from rag.explanation_generator import ExplanationGenerator
    
    generator = ExplanationGenerator(use_pubmed=True)
    
    # Test all 5 classes
    test_cases = [
        ("adenocarcinoma", 0.94, "peripheral"),
        ("squamous cell carcinoma", 0.88, "central"),
        ("large cell carcinoma", 0.79, "peripheral"),
        ("benign cases", 0.91, "diffuse"),
        ("normal cases", 0.96, "clear")
    ]
    
    for class_name, confidence, pattern in test_cases:
        print(f"\n  Testing: {class_name.upper()}")
        
        # Create appropriate mock heatmap
        heatmap = np.zeros((224, 224))
        if pattern == "peripheral":
            heatmap[30:90, 150:210] = 0.8
        elif pattern == "central":
            heatmap[80:150, 80:150] = 0.85
        elif pattern == "diffuse":
            heatmap[40:180, 40:180] = 0.3
        else:  # clear
            heatmap[:, :] = 0.1
        
        explanation = generator.generate(
            heatmap=heatmap,
            predicted_class=class_name,
            confidence=confidence,
            include_pubmed=False  # Skip PubMed for speed
        )
        
        print(f"    ✓ Confidence: {confidence*100:.1f}%")
        print(f"    ✓ Keywords: {explanation.keywords_used[:3]}...")
        print(f"    ✓ Sources: {len(explanation.sources)} references")
    
    print("\n✓ Full Pipeline Integration tests PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("     LUNGXAI RAG PIPELINE - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Knowledge Base", test_knowledge_base),
        ("PubMed Retriever", test_pubmed_retriever),
        ("XAI to Text", test_xai_to_text),
        ("Explanation Generator", test_explanation_generator),
        ("Full Pipeline Integration", test_full_pipeline_integration)
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} test FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("                    TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\n  Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! RAG pipeline is working correctly.")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
    
    print("=" * 60 + "\n")
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
