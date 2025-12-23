# =============================================================================
# SRC PACKAGE
# Explainable AI for Multi-Class Lung Cancer Classification
# =============================================================================
"""
Main source package containing all modules for the lung cancer classification system.

Package Structure:
    - data/     : Dataset loading, preprocessing, augmentation
    - models/   : Neural network architectures (ResNet-50, etc.)
    - xai/      : Explainable AI implementations (Grad-CAM)
    - rag/      : RAG pipeline for generating explanations
    - utils/    : Helper functions, configuration, metrics

WHY THIS STRUCTURE?
    1. Separation of Concerns: Each module has a single responsibility
    2. Testability: Each component can be tested independently
    3. Reusability: Components can be reused across different experiments
    4. Academic Clarity: Easy to explain each module during viva
"""

__version__ = "1.0.0"
__author__ = "Major Project Team"
