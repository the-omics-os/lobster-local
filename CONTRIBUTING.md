# 🤝 Contributing to Lobster AI

Thank you for your interest in contributing to Lobster AI! We welcome contributions from the bioinformatics and AI communities.

## 🌟 Ways to Contribute

### 🐛 Bug Reports
- Use [GitHub Issues](https://github.com/homara-ai/lobster/issues) with the "bug" label
- Include steps to reproduce, expected vs actual behavior
- Add relevant system information (OS, Python version)

### 💡 Feature Requests
- Open an issue with the "enhancement" label
- Describe the use case and benefits
- Consider if it fits the core bioinformatics mission

### 📝 Documentation
- Fix typos, improve clarity, add examples
- Update installation guides
- Create tutorial content

### 🔬 Code Contributions
- New bioinformatics analysis methods
- Performance improvements
- Test coverage enhancements
- Bug fixes

## 🚀 Development Setup

### Prerequisites
- Python 3.12+
- Git
- Virtual environment (recommended)

### Setup Process
```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/lobster.git
cd lobster

# 3. Set up development environment  
make dev-install

# 4. Create a branch for your changes
git checkout -b feature/your-feature-name

# 5. Make your changes and test
make test
make format  # Auto-format code
make lint    # Check code quality

# 6. Commit and push
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name

# 7. Create Pull Request on GitHub
```

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for all functions
- Add comprehensive docstrings
- Keep functions focused and testable

### Testing
- Add tests for new functionality
- Ensure existing tests pass
- Test with real bioinformatics data when possible
- Include edge cases and error conditions

### Bioinformatics Focus
- Prioritize scientific accuracy over performance
- Include literature references for methods
- Follow established bioinformatics conventions
- Consider reproducibility in all analyses

### Documentation
- Update README.md if adding major features
- Add docstrings to all public functions
- Include examples for complex functionality
- Update INSTALLATION.md for setup changes

## 🏗️ Architecture Overview

### Core Components
- **lobster/agents/**: AI agents for different analysis types
- **lobster/core/**: Data management and client infrastructure  
- **lobster/tools/**: Analysis services and bioinformatics tools
- **lobster/config/**: Configuration management
- **lobster-core/**: Shared interfaces and utilities

### Key Design Principles
1. **Modular**: Each component has clear responsibilities
2. **Extensible**: Easy to add new analysis methods
3. **Testable**: Services are stateless and unit testable
4. **User-friendly**: Natural language interface
5. **Reproducible**: Complete provenance tracking

## 🔬 Adding New Analysis Methods

### Example: Adding a New Tool
```python
# lobster/tools/new_analysis_service.py
from typing import Dict, Any, Tuple
import anndata

class NewAnalysisService:
    """Service for new bioinformatics analysis."""
    
    def analyze_data(
        self,
        adata: anndata.AnnData,
        parameter1: float = 0.5,
        parameter2: str = "default"
    ) -> Tuple[anndata.AnnData, Dict[str, Any]]:
        """
        Perform new analysis on AnnData.
        
        Args:
            adata: Input AnnData object
            parameter1: Description of parameter
            parameter2: Another parameter
            
        Returns:
            Tuple of (processed_adata, analysis_statistics)
        """
        # 1. Create working copy
        adata_processed = adata.copy()
        
        # 2. Perform analysis
        # ... your analysis logic here ...
        
        # 3. Return results with statistics
        stats = {
            "analysis_type": "new_analysis",
            "parameters_used": {"parameter1": parameter1, "parameter2": parameter2},
            "results_summary": "Analysis completed successfully"
        }
        
        return adata_processed, stats
```

### Adding Agent Tool Integration
```python
# In relevant agent file (e.g., lobster/agents/transcriptomics_expert.py)
@tool
def run_new_analysis(
    modality_name: str,
    parameter1: float = 0.5,
    parameter2: str = "default"
) -> str:
    """Run new analysis on specified modality."""
    try:
        # Get data
        adata = data_manager.get_modality(modality_name)
        
        # Run analysis
        service = NewAnalysisService()
        result_adata, stats = service.analyze_data(adata, parameter1, parameter2)
        
        # Save results
        new_modality = f"{modality_name}_new_analysis"
        data_manager.modalities[new_modality] = result_adata
        
        # Log operation
        data_manager.log_tool_usage(
            tool_name="run_new_analysis",
            parameters={"parameter1": parameter1, "parameter2": parameter2},
            description=f"Applied new analysis to {modality_name}"
        )
        
        return f"✅ New analysis complete! Results saved as '{new_modality}'"
        
    except Exception as e:
        return f"❌ Analysis failed: {str(e)}"
```

## ✅ Pull Request Process

### Before Submitting
- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] No linting errors (`make lint`)
- [ ] Documentation updated
- [ ] Example/test included for new features

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Breaking change

## Testing
- [ ] Tests added/updated
- [ ] Manual testing performed
- [ ] Example data tested

## Bioinformatics Impact
- [ ] Scientifically accurate
- [ ] Follows established conventions
- [ ] Literature references included (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## 🧪 Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_specific.py -v

# Run with coverage
make test-coverage

# Run integration tests
make test-integration
```

## 💬 Community & Support

### Getting Help
- 💬 **[Discord Community](https://discord.gg/homaraai)** - Chat with other contributors
- 📚 **[Documentation](docs/)** - Comprehensive guides
- 🐛 **[GitHub Issues](https://github.com/homara-ai/lobster/issues)** - Bug reports and feature requests

### Code of Conduct
We follow a simple principle: **Be kind, be constructive, be scientific.**

- Respect all contributors regardless of experience level
- Focus on facts and scientific accuracy
- Help newcomers learn and contribute
- Give constructive feedback on code and ideas

## 📄 License

By contributing to Lobster AI, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

All contributors are recognized in our [AUTHORS.md](AUTHORS.md) file and release notes.

---

**Ready to contribute? We can't wait to see what you build! 🦞**
