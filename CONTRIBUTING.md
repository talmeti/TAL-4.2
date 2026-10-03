# Contributing to TAL

Thanks for your interest in TAL! This document explains how to
contribute bug reports, feature requests, and code.

## Reporting bugs

Please open an issue on GitHub and include:

- a minimal reproducible example (function + seed + expected/observed output),
- the exact versions of Python, NumPy, SymPy, lmfit, and joblib,
- the output of `python tal_4_2_step03_reproducibility.py` if relevant.

## Requesting features

Open an issue with the label `enhancement`. Describe the use case,
not just the implementation. If possible, sketch the API you would like.

## Contributing code

1. Fork the repository and create a branch named `feature/<short-name>`.
2. Follow the existing code style (PEP 8, type hints, docstrings).
3. Add tests for any new behavior in `tal_4_2_step02_tests.py`.
4. Run the full test suite locally:

       pytest tal_4_2_step02_tests.py -v

5. Update `CHANGELOG.md`.
6. Open a pull request with a clear description and link to the issue.

## Development setup

    conda env create -f environment.yml
    conda activate tal
    pip install -e .[dev]

## Code of conduct

By participating in this project, you agree to abide by the
`CODE_OF_CONDUCT.md`.

