# Contributing to TruthLens AI

Thank you for considering contributing to TruthLens AI! 🎉

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Error logs if applicable

### Suggesting Features

1. Check existing issues for similar suggestions
2. Create a new issue with:
   - Clear description of the feature
   - Use case and benefits
   - Possible implementation approach

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Format code with black (`black .`)
7. Commit changes (`git commit -m 'Add amazing feature'`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Code Style

- Follow PEP 8
- Use type hints where applicable
- Add docstrings for functions and classes
- Keep functions focused and small
- Write meaningful commit messages

### Testing

- Write tests for new features
- Ensure existing tests pass
- Aim for >80% code coverage
- Test edge cases

### Documentation

- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update config files if needed

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/truthlens-ai.git
cd truthlens-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .
```

## Code Review Process

1. Maintainers will review your PR
2. Address any feedback
3. Once approved, PR will be merged
4. Your contribution will be credited!

## Questions?

Feel free to open an issue for any questions!

Thank you for contributing! 🙏
