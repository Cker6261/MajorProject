# 📚 LungXAI Documentation

This directory contains comprehensive documentation for the LungXAI project.

## 📋 Document Overview

| Document | Description | Audience |
|----------|-------------|----------|
| [PROJECT_REVIEW_GUIDE.md](PROJECT_REVIEW_GUIDE.md) | Complete project documentation for defense/viva | Academic review |
| [PSEUDOCODE.md](PSEUDOCODE.md) | Algorithm pseudocode and technical details | Developers |
| [LungXAI_Research_Paper.md](LungXAI_Research_Paper.md) | IEEE-style research paper | Academic publication |
| [BASELINE_VS_FINETUNED_COMPARISON.md](BASELINE_VS_FINETUNED_COMPARISON.md) | Transfer learning vs from-scratch comparison | Researchers |

## 🗂️ Directory Structure

```
docs/
├── README.md                              # This file
├── PROJECT_REVIEW_GUIDE.md                # Complete project documentation
├── PSEUDOCODE.md                          # Algorithm pseudocode
├── LungXAI_Research_Paper.md              # Research paper (source)
├── BASELINE_VS_FINETUNED_COMPARISON.md    # Model comparison study
├── pipeline_diagram.py                    # Script to generate architecture diagram
├── pipeline_pseudocode.py                 # Script with code examples
├── create_ieee_paper.py                   # Script to generate IEEE paper
├── convert_to_pdf.py                      # PDF conversion utility
└── images/                                # Documentation images
    ├── architecture.png                   # System architecture diagram
    ├── dataset_dist.png                   # Dataset distribution chart
    └── gradcam_flow.png                   # Grad-CAM flow diagram
```

## 📖 Quick Links

### For Project Review/Defense
1. Start with [PROJECT_REVIEW_GUIDE.md](PROJECT_REVIEW_GUIDE.md) - contains everything for viva
2. Review [PSEUDOCODE.md](PSEUDOCODE.md) for algorithm explanations

### For Technical Understanding
1. [PSEUDOCODE.md](PSEUDOCODE.md) - Step-by-step algorithms
2. [BASELINE_VS_FINETUNED_COMPARISON.md](BASELINE_VS_FINETUNED_COMPARISON.md) - Model analysis

### For Academic Publication
1. [LungXAI_Research_Paper.md](LungXAI_Research_Paper.md) - IEEE format paper

## 🔧 Generating PDFs

To generate PDF versions of documentation:

```bash
# Using the conversion script
python docs/convert_to_pdf.py

# Or using pandoc directly
pandoc docs/PROJECT_REVIEW_GUIDE.md -o docs/PROJECT_REVIEW_GUIDE.pdf
```

## 📷 Images

The `images/` directory contains diagrams and visualizations:

- **architecture.png**: Complete system architecture
- **dataset_dist.png**: Dataset class distribution
- **gradcam_flow.png**: Grad-CAM explanation flow

## 📝 Notes

- All markdown files are the source of truth
- PDF/DOCX files can be regenerated from markdown
- Images should be relative to the docs/ directory
- Keep documentation up-to-date with code changes

## 🔗 Related Documentation

- [../README.md](../README.md) - Main project README
- [../COMMANDS.md](../COMMANDS.md) - Command reference
- [../CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [../CHANGELOG.md](../CHANGELOG.md) - Version history