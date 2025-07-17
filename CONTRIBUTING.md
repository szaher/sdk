# Contributing to the Kubeflow SDK

Thank you for your interest in contributing to the Kubeflow SDK!

## Getting Started

### Prerequisites
- Python 3.9–3.11
- [pip](https://pip.pypa.io/en/stable/)
- [pre-commit](https://pre-commit.com/)
- uv

### Setting Up Your Development Environment
Clone the repository:
```sh
git clone https://github.com/kubeflow/sdk.git
cd sdk
```

Install uv if not installed [Official Docs](https://docs.astral.sh/uv/getting-started/installation/) or using the following command
```sh
make uv
```
### Install SDK & Dependencies
Use uv to create a virtualenv if not created and install dependencies
```sh
uv sync
```

Install development tools:
```sh
uv sync --dev
```

## Development Workflow

### Pre-commit
We use pre-commit to ensure consistent code formatting. To enable pre-commit hooks, run:
```sh
pre-commit install
```
To run all hooks manually:
```sh
pre-commit run --all-files
```

### Testing
To run the unit tests (if present), execute:
```sh
pytest
```

### Code Coverage
To run tests and measure coverage:
```sh
coverage run -m pytest
coverage report -m
```

### Code Formatting
To check formatting:
```shell
make verify 
```

#### Using Ruff

```shell
uvx ruff check --show-fixes
```

To auto-format, lint all files:

```shell
uvx ruff check --fix
```

## Continuous Integration
All PRs are automatically checked by CI. Please ensure all checks pass before requesting review.

## Getting Help
For questions, open an issue or contact a maintainer listed in `OWNERS`.

## Resources
- [Kubeflow Trainer Docs](https://www.kubeflow.org/docs/components/trainer/)
- [Source Code](https://github.com/kubeflow/trainer)

---
