# Changelog

All notable changes to the LungXAI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-22

### 🎉 Initial Release

This is the first official release of LungXAI - Explainable AI for Lung Cancer Classification.

### Added

#### Core Features
- **Multi-Model Classification Pipeline**: Support for 4 deep learning architectures
  - ResNet-50 (96.97% accuracy)
  - MobileNetV2 (97.40% accuracy)
  - Vision Transformer ViT-B/16 (93.51% accuracy)
  - Swin Transformer Tiny (97.84% accuracy) - **Best performing**

- **Explainable AI (XAI) Module**
  - Grad-CAM implementation for visual explanations
  - Heatmap overlay generation showing model focus areas
  - XAI-to-text bridge for extracting visual features

- **RAG-Based Knowledge Retrieval**
  - Medical knowledge base with lung cancer information
  - PubMed API integration for research paper retrieval
  - Context-aware explanation generation

- **Training Infrastructure**
  - Multi-model training with caching support
  - Automatic checkpoint management
  - Training history logging and visualization
  - Model comparison and benchmarking tools

- **Demo and Inference Tools**
  - `demo_multi_model.py` - Interactive multi-model demo
  - `demo.py` - Visual demo with Grad-CAM
  - Model comparison visualization
  - Command-line interface for all operations

#### Documentation
- Comprehensive README with installation guide
- Project Review Guide for academic defense
- Command reference (COMMANDS.md)
- Research paper documentation
- Pseudocode and algorithm documentation
- Contributing guidelines

#### Dataset Support
- 5-class lung cancer classification
  - Adenocarcinoma
  - Squamous Cell Carcinoma
  - Large Cell Carcinoma
  - Benign Cases
  - Normal Cases
- Data augmentation pipeline
- Train/validation/test split utilities

### Technical Details

#### Model Performance Summary
| Model | Accuracy | F1-Score | Parameters |
|-------|----------|----------|------------|
| Swin-T | 97.84% | 97.84% | 28M |
| MobileNetV2 | 97.40% | 97.40% | 3.5M |
| ResNet-50 | 96.97% | 96.95% | 25.6M |
| ViT-B/16 | 93.51% | 93.48% | 86M |

#### Dependencies
- PyTorch 2.0+
- torchvision 0.15+
- Python 3.8+
- CUDA support (optional but recommended)

---

## [0.9.0] - 2026-01-20

### Added
- Baseline vs Fine-tuned model comparison study
- Training from scratch implementation
- Comparison documentation and analysis

### Changed
- Improved model factory with better architecture support
- Enhanced configuration management

---

## [0.8.0] - 2026-01-15

### Added
- Swin Transformer model support
- Vision Transformer (ViT) integration
- Multi-model training script (`train_all_models.py`)
- Model comparison tools

### Changed
- Refactored model factory for cleaner architecture selection
- Improved caching mechanism for trained models

---

## [0.7.0] - 2025-12-20

### Added
- MobileNetV2 model for edge deployment
- RAG pipeline with PubMed integration
- Explanation generator module

### Fixed
- Memory management for GPU training
- Cache directory handling

---

## [0.6.0] - 2025-12-10

### Added
- Grad-CAM implementation
- Visualization utilities
- XAI-to-text bridge module

---

## [0.5.0] - 2025-12-01

### Added
- Initial project structure
- ResNet-50 baseline classifier
- Data loading and preprocessing
- Basic training pipeline

---

## Future Plans

### [1.1.0] - Planned
- [ ] Semantic RAG with sentence transformers
- [ ] Additional XAI methods (LIME, SHAP)
- [ ] Web interface
- [ ] Docker containerization

### [1.2.0] - Planned
- [ ] Model ensemble predictions
- [ ] Real-time inference optimization
- [ ] Mobile deployment support

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.