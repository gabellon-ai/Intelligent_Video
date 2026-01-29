# Contributing

Thanks for your interest in contributing to Intelligent Video!

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install dev tools: `pip install ruff pytest`

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting. Run before committing:

```bash
ruff check src/
ruff format src/
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Ensure CI passes (linting + import tests)
4. Submit PR with clear description

## Adding Detection Targets

To add new objects for detection:

1. Add descriptive text queries to the `queries` list
2. Test with representative images
3. Tune confidence threshold if needed
4. Document the new target in README.md

## Reporting Issues

Please include:
- Python version
- GPU model (if applicable)
- Full error traceback
- Sample image (if relevant)
