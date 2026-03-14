# coreason_etl_snomed

ETL process for integrating SNOMED CT clinical healthcare terminology data

[![CI/CD](https://github.com/CoReason-AI/coreason_etl_snomed/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/CoReason-AI/coreason_etl_snomed/actions/workflows/ci-cd.yml)
[![PyPI](https://img.shields.io/pypi/v/coreason_etl_snomed.svg)](https://pypi.org/project/coreason_etl_snomed/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/coreason_etl_snomed.svg)](https://pypi.org/project/coreason_etl_snomed/)
[![License](https://img.shields.io/github/license/CoReason-AI/coreason_etl_snomed)](https://github.com/CoReason-AI/coreason_etl_snomed/blob/main/LICENSE)
[![Codecov](https://codecov.io/gh/CoReason-AI/coreason_etl_snomed/branch/main/graph/badge.svg)](https://codecov.io/gh/CoReason-AI/coreason_etl_snomed)
[![Downloads](https://static.pepy.tech/badge/coreason_etl_snomed)](https://pepy.tech/project/coreason_etl_snomed)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## Getting Started

### Prerequisites

- Python 3.14+
- uv

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/CoReason-AI/coreason_etl_snomed.git
    cd coreason_etl_snomed
    ```
2.  Install dependencies:
    ```sh
    uv sync --all-extras --dev
    ```

### Usage

-   Run the linter:
    ```sh
    uv run pre-commit run --all-files
    ```
-   Run the tests:
    ```sh
    uv run pytest
    ```
