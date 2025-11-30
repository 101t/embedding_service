# Contributing to FastEmbed Service

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/101t/embedding_service.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run checks: `make check`
6. Commit your changes: `git commit -m "feat: add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

### Prerequisites

- Rust 1.75 or later
- Docker (optional, for containerized development)

### Building

```bash
# Debug build
make build

# Release build
make release

# Run locally
make run
```

### Running Tests

```bash
make test
```

### Code Style

We use standard Rust formatting and linting:

```bash
# Format code
make fmt

# Run linter
make lint

# Run all checks
make check
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add support for new embedding model
fix: resolve memory leak in batch processing
docs: update API documentation
```

## Pull Request Guidelines

1. **Keep PRs focused** - One feature or fix per PR
2. **Write tests** - Add tests for new functionality
3. **Update documentation** - Update README if needed
4. **Pass CI checks** - Ensure all checks pass before requesting review

## Reporting Issues

When reporting issues, please include:

1. A clear description of the problem
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details (OS, Rust version, etc.)
5. Relevant logs or error messages

## Adding New Models

To add support for a new embedding model:

1. Check if the model is supported by [fastembed-rs](https://github.com/Anush008/fastembed-rs)
2. Add the model mapping in `src/main.rs`
3. Update the README with model information
4. Add tests for the new model

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
