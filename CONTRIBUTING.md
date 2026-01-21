# 🤝 Contributing to LungXAI

First off, thank you for considering contributing to LungXAI! It's people like you that make this project a great tool for the medical AI research community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Community](#community)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our commitment to fostering an open, welcoming, diverse, and inclusive community. By participating, you are expected to uphold these values.

### Our Standards

- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome newcomers and help them get involved
- **Be collaborative**: Work together towards common goals
- **Be constructive**: Provide helpful feedback and suggestions
- **Academic integrity**: Respect intellectual property and give proper attribution

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ 
- Git
- Familiarity with PyTorch, computer vision, or medical AI (helpful but not required)
- Understanding of machine learning concepts

### First Contribution

1. **Look for beginner-friendly issues** labeled with `good first issue` or `help wanted`
2. **Read the documentation** thoroughly, especially the [Project Review Guide](docs/PROJECT_REVIEW_GUIDE.md)
3. **Set up your development environment** following the instructions below
4. **Start small** - documentation improvements, bug fixes, or small features

## 🛠️ How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples and expected behavior**
- **Include your environment details** (OS, Python version, PyTorch version)
- **Add relevant logs or error messages**

### ✨ Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a step-by-step description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **Include mockups or examples** if applicable

### 📝 Documentation Improvements

Documentation improvements are always welcome! This includes:

- Fixing typos or grammatical errors
- Clarifying confusing sections
- Adding examples or tutorials
- Updating outdated information
- Translating content

### 🧬 Code Contributions

Areas where we especially welcome contributions:

#### High Priority
- **Model improvements**: New architectures, optimization techniques
- **Explainability methods**: Additional XAI techniques (LIME, SHAP, etc.)
- **RAG enhancements**: Better knowledge retrieval and explanation generation
- **Testing**: Unit tests, integration tests, model validation

#### Medium Priority
- **Data handling**: New datasets, data augmentation techniques
- **Performance optimization**: Speed improvements, memory efficiency
- **Visualization**: Better plots, interactive dashboards
- **Mobile/Edge deployment**: Model quantization, ONNX conversion

#### Research Contributions
- **Novel algorithms**: New explainability techniques
- **Medical domain knowledge**: Clinical insights, domain-specific features
- **Evaluation metrics**: Medical-specific evaluation criteria
- **Benchmarking**: Comparison with other medical AI systems

## 🛠️ Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/LungXAI.git
cd LungXAI

# Add the original repository as upstream
git remote add upstream https://github.com/originalusername/LungXAI.git
```

### 2. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available
```

### 3. Development Dependencies

```bash
# Install additional development tools
pip install black isort flake8 pytest pytest-cov mypy
pip install pre-commit  # For automated code formatting
```

### 4. Pre-commit Setup (Recommended)

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files (optional)
pre-commit run --all-files
```

### 5. Download Test Data

```bash
# Download a small subset of data for testing
# Follow instructions in README.md for full dataset
```

### 6. Verify Installation

```bash
# Run basic tests
python -m pytest tests/ -v

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"

# Quick functionality test
python demo_multi_model.py --list
```

## 📐 Style Guidelines

### Python Code Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use `isort` for automatic import sorting
- **Formatting**: Use `Black` for automatic code formatting
- **Type hints**: Encouraged, especially for public functions
- **Docstrings**: Google style docstrings

#### Example Function

```python
def predict_image(
    image_path: str, 
    model_name: str = "resnet50"
) -> Dict[str, Any]:
    """
    Predict lung cancer type from CT scan image.
    
    Args:
        image_path: Path to the CT scan image file
        model_name: Name of the model to use for prediction
        
    Returns:
        Dictionary containing prediction results and explanations
        
    Raises:
        FileNotFoundError: If image_path doesn't exist
        ModelNotFoundError: If model_name is not available
    """
    # Implementation here
    pass
```

### Documentation Style

- **Markdown**: Use proper Markdown formatting
- **Code blocks**: Include language specification
- **Images**: Use relative paths, store in appropriate directories
- **Links**: Check that all links work
- **Headers**: Use consistent header hierarchy

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix  
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(models): add Swin Transformer v2 support
fix(gradcam): resolve memory leak in batch processing
docs(readme): update installation instructions
test(pipeline): add unit tests for ExplainablePipeline
```

## 📤 Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** following the style guidelines

4. **Add/update tests** for any new functionality

5. **Update documentation** as needed

6. **Run tests locally**:
   ```bash
   python -m pytest tests/ -v
   python -m flake8 src/
   python -m black --check src/
   ```

### Submitting Your PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub with:
   - **Clear title** describing the change
   - **Detailed description** of what you did and why
   - **Reference any related issues** using `#issue-number`
   - **Include screenshots** for UI changes
   - **List any breaking changes**

### PR Review Process

1. **Automated checks** must pass (if CI/CD is set up)
2. **Code review** by project maintainers
3. **Address feedback** promptly and professionally  
4. **Squash commits** if requested before merging

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (describe)

## Testing
- [ ] Existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or described above)
```

## 🐛 Issue Guidelines

### Issue Types

Use appropriate labels:
- **bug**: Something isn't working
- **enhancement**: New feature request
- **documentation**: Documentation improvements
- **good first issue**: Good for newcomers
- **help wanted**: Extra attention needed
- **question**: Questions about usage
- **research**: Research-related discussions

### Issue Templates

We provide templates for:
- Bug reports
- Feature requests
- Documentation improvements
- Research collaborations

## 👥 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Pull Requests**: Code review and collaboration

### Getting Help

- **Documentation**: Check the [docs/](docs/) directory first
- **Examples**: Look at existing code in [src/](src/) directory
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Ask questions in GitHub Discussions

### Recognition

Contributors will be:
- **Listed** in the contributors section of README.md
- **Mentioned** in release notes for significant contributions
- **Credited** in academic publications if substantial research contributions

## 🎓 Academic Contributions

### Research Collaboration

We welcome research collaborations! If you're:
- **Student researcher**: Working on related projects or thesis
- **Faculty member**: Interested in collaboration or extensions
- **Industry researcher**: Exploring medical AI applications

Please reach out via GitHub Issues or Discussions.

### Publications and Citations

- **Using this work**: Please cite properly (see README.md)
- **Extending this work**: We're happy to collaborate on publications
- **Academic integrity**: Always maintain proper attribution

## 🏆 Recognition

### Hall of Fame

We recognize outstanding contributors in our project documentation and README.md.

### Contribution Levels

- **🌟 First-time contributor**: Made their first contribution
- **🚀 Regular contributor**: Multiple quality contributions
- **🧬 Core contributor**: Significant ongoing contributions
- **🏆 Maintainer**: Project maintenance and leadership

## ❓ Questions?

Don't hesitate to ask! We're here to help:

- **General questions**: GitHub Discussions
- **Specific issues**: GitHub Issues
- **Private matters**: Contact maintainers via email

---

## 🙏 Thank You!

Every contribution, no matter how small, makes a difference. We appreciate your time and effort in helping improve LungXAI!

**Happy Contributing! 🎉**