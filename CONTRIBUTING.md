# Contributing to Lobster AI 🦞

Thank you for your interest in contributing to Lobster AI! We welcome contributions from the community and are grateful for your help in making this project better.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/your-username/lobster-ai.git
cd lobster-ai
```

2. **Set up development environment**

```bash
# Install with development dependencies
make dev-install

# Or manually:
pip install -e ".[dev]"
pre-commit install
```

3. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run tests to verify setup**

```bash
make test
```

## 🛠️ Development Workflow

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pylint**: Advanced linting
- **bandit**: Security checking

Format your code before committing:

```bash
make format
make lint
```

### Testing

We use pytest for testing. Write tests for new features and bug fixes:

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_agents.py -v

# Run with coverage
make test
```

Test categories:
- **Unit tests**: Fast, isolated tests for individual functions
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Commit Messages

Use conventional commit messages:

```
feat: add new clustering algorithm
fix: resolve GEO download timeout issue  
docs: update installation guide
test: add unit tests for PubMed service
refactor: optimize agent communication
```

## 📝 Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:

- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

Use our bug report template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run `lobster chat`
2. Enter query: "..."
3. See error

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., macOS 13.0]
- Python: [e.g., 3.11.0]  
- Lobster AI: [e.g., 1.0.0]
```

### ✨ Feature Requests

For new features:

- Describe the problem you're solving
- Propose a solution
- Consider alternative solutions
- Think about backward compatibility

### 🔧 Code Contributions

#### Areas where we need help:

1. **New Analysis Tools**
   - Additional bioinformatics algorithms
   - New visualization types
   - Data format support

2. **Agent Improvements**
   - New specialized agents
   - Better prompts and reasoning
   - Performance optimizations

3. **Infrastructure** 
   - Documentation improvements
   - CI/CD enhancements
   - Docker optimizations

4. **Testing**
   - More comprehensive test coverage
   - Performance benchmarks
   - Integration tests

#### Pull Request Process

1. **Create a feature branch**

```bash
git checkout -b feature/add-new-algorithm
```

2. **Make your changes**
   - Write code following our style guide
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**

```bash
make test
make lint
make type-check
```

4. **Commit and push**

```bash
git add .
git commit -m "feat: add new clustering algorithm"
git push origin feature/add-new-algorithm
```

5. **Create Pull Request**
   - Use our PR template
   - Link related issues
   - Describe changes and rationale
   - Add screenshots for UI changes

### 📚 Documentation

Help improve our documentation:

- Fix typos and grammar
- Add examples and tutorials
- Improve API documentation
- Create video guides

Documentation is in the `docs/` directory and uses Markdown.

## 🧬 Bioinformatics Guidelines

### Data Handling

- Always validate input data
- Handle missing values appropriately
- Use appropriate data structures (AnnData for single-cell)
- Document expected data formats

### Algorithm Implementation

- Follow established bioinformatics practices
- Cite relevant papers in docstrings
- Include parameter validation
- Provide sensible defaults

### Visualization

- Use colorblind-friendly palettes
- Include proper axis labels and legends
- Support both interactive and static formats
- Follow publication quality standards

## 🤖 AI Agent Development

### Creating New Agents

1. **Agent Structure**

```python
# lobster/agents/new_agent.py
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

def new_agent(data_manager, callback_handler=None, agent_name="new_agent"):
    # Define tools
    @tool
    def tool_function(query: str) -> str:
        """Tool description."""
        # Implementation
        pass
    
    # Create agent with tools and prompt
    return create_react_agent(
        model=llm,
        tools=[tool_function],
        prompt=system_prompt,
        name=agent_name
    )
```

2. **Register in Graph**

Add your agent to `lobster/agents/graph.py`.

3. **Add Tests**

Create tests for your agent in `tests/test_agents.py`.

### Tool Development

Tools should be:
- **Focused**: Do one thing well
- **Robust**: Handle errors gracefully  
- **Documented**: Clear docstrings
- **Tested**: Unit tests included

## 🔬 Research Contributions

We welcome research-oriented contributions:

- Novel algorithms for bioinformatics
- Benchmarking studies
- Performance optimizations
- Integration of new models/APIs

Please include:
- Literature review and citations
- Experimental validation
- Performance comparisons
- Documentation of methodology

## 📊 Performance Guidelines

### Optimization Priorities

1. **Memory efficiency** for large datasets
2. **Computation speed** for interactive use
3. **Scalability** for production environments

### Benchmarking

- Include performance tests for new features
- Document time/space complexity
- Test with realistic data sizes
- Compare against existing methods

## 🌍 Community

### Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/). Be respectful, inclusive, and professional.

### Getting Help

- 💬 **Discord**: [Join our community](https://discord.gg/homaraai)
- 📧 **Email**: support@homara.ai
- 🐛 **Issues**: GitHub Issues for bugs and features

### Recognition

Contributors are recognized in:
- Release notes
- Contributors section
- Special thanks in major releases

## 📋 Release Process

For maintainers:

1. **Version Bump**

```bash
make bump-minor  # or bump-patch, bump-major
```

2. **Update Changelog**

Document changes in `CHANGELOG.md`.

3. **Create Release**

```bash
make release
```

4. **Publish**

```bash
make publish
```

---

Thank you for contributing to Lobster AI! Your help makes bioinformatics more accessible to researchers worldwide. 🦞🧬
