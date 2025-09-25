# ===== CONTRIBUTING.md =====
# Contributing to STL-PINN Processor

Thank you for your interest in contributing to STL-PINN Processor! This document provides guidelines and information for contributors.

## 🤝 Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git
- Docker (for containerized development)
- NVIDIA GPU (optional, for PINN training)

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/stl-pinn-processor.git
   cd stl-pinn-processor
   ```

3. Run the setup script:
   ```bash
   ./scripts/setup.sh
   ```

4. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## 📝 Development Workflow

### Branch Naming

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Add tests for new functionality

4. Run the test suite:
   ```bash
   ./scripts/run-tests.sh
   ```

5. Commit your changes:
   ```bash
   git commit -m "Add: descriptive commit message"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a Pull Request

### Commit Message Guidelines

Use conventional commits:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## 🏗️ Architecture Guidelines

### Code Organization

```
src/
├── config/          # Configuration management
├── core/           # Core abstractions and interfaces
├── services/       # Business logic services
├── models/         # Data models and schemas
├── utils/          # Utility functions
└── api/            # API layer
```

### Design Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **Dependency Injection**: Use dependency injection for testability
3. **Async First**: Prefer async/await for I/O operations
4. **Type Safety**: Use type hints throughout
5. **Error Handling**: Comprehensive error handling and logging

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

1. **Unit Tests**: Test individual functions/methods in isolation
2. **Integration Tests**: Test component interactions
3. **Mock External Dependencies**: Use mocks for external services
4. **Test Edge Cases**: Include error conditions and boundary cases

### Test Coverage

- Maintain >80% code coverage
- All new features must include tests
- Bug fixes must include regression tests

## 📊 Code Quality

### Linting and Formatting

We use several tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance considerations addressed
- [ ] Error handling is comprehensive

## 📚 Documentation

### Documentation Standards

1. **Docstrings**: Use Google-style docstrings
2. **Type Hints**: Include type hints for all functions
3. **Comments**: Explain complex logic, not obvious code
4. **README Updates**: Update documentation for new features

### API Documentation

- Use FastAPI's automatic documentation
- Include request/response examples
- Document error codes and messages

## 🚢 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Breaking changes increment MAJOR
- New features increment MINOR
- Bug fixes increment PATCH

### Release Checklist

1. Update CHANGELOG.md
2. Update version in setup.py
3. Create release tag
4. Build and test Docker images
5. Deploy to staging
6. Create GitHub release
7. Update documentation

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Reproduction**: Step-by-step reproduction instructions
3. **Environment**: OS, Python version, dependencies
4. **Expected vs Actual**: What should happen vs what happens
5. **Logs**: Relevant error messages or logs
6. **Files**: Sample files if applicable

## 💡 Feature Requests

For feature requests:

1. **Use Case**: Describe the specific use case
2. **Benefit**: Explain how it helps users
3. **Implementation**: Suggest implementation approach
4. **Alternatives**: Consider alternative solutions

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### Core Features
- New mesh processing algorithms
- Additional PINN physics models
- Advanced LLM integrations
- Performance optimizations

### Quality Improvements
- Test coverage improvements
- Documentation updates
- Bug fixes
- Security enhancements

### Integrations
- New CAD format support
- Additional LLM providers
- Cloud platform integrations
- Monitoring and observability

### Research
- Novel PINN architectures
- Advanced mesh quality metrics
- Machine learning optimizations
- Physics simulation improvements

## 🤖 AI and LLM Contributions

Special guidelines for AI/LLM related contributions:

1. **Model Selection**: Justify choice of AI models
2. **Prompt Engineering**: Document prompt design decisions
3. **Evaluation**: Include evaluation metrics and benchmarks
4. **Bias Considerations**: Address potential biases
5. **Fallback Strategies**: Implement fallbacks for AI failures

## 📞 Getting Help

- **GitHub Discussions**: For questions and discussions
- **GitHub Issues**: For bugs and feature requests
- **Email**: contact@stl-pinn-processor.com for private matters

## 🙏 Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes for significant contributions
- Invited to join the core team for ongoing contributors

Thank you for contributing to STL-PINN Processor!
