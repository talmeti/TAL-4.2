# Changelog

All notable changes to TAL are documented in this file.
The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

## [4.2.0] - 2026-10-01

### Added

- Adaptive selection between base grammar (8 operators) and extended
  grammar (13 operators).
- Parametric memory: reuse of fitted parameter vectors across trees
  sharing the same structural signature.
- Deterministic acceptance criterion based on relative MSE improvement.
- Limit detector for residual structure analysis.
- Multi-start LM fitter with warm starts from memory.
- 41-test pytest suite covering grammars, utilities, memory, fitter,
  theory, acceptance, detector, generator, domains, and two end-to-end
  cases.
- Reproducibility report script.
- JOSS submission files (paper.md, paper.bib, community guidelines).
- CI workflow for automated testing on GitHub Actions.

### Changed

- Renamed from TEA to TAL (Theory-Adaptive Learning).
- Refactored tree cloning to use a local RNG for reproducibility.
- Removed unused imports and dead code.

### Fixed

- Non-determinism in tree cloning (now fully reproducible per seed).
- Verbose output redirection (Tee) for console and file logging.

