# TAL 4.2 — Report di benchmark

**Data**: 2026-10-03
**Autore**: Ruggero Fenech - Giorgio Casoni
**Versione**: 1.0

## Panoramica

TAL 4.2 (Theory-Adaptive Learning) e' un metodo di symbolic regression
con grammatiche adattive, confrontato con poly e PySR.

## Risultati principali

| Metodo | n | MSE median | Solved rate |
|--------|---|------------|-------------|
| poly | 51 | 4.32e-04 | 60.78% |
| PySR | 51 (44 ok) | 1.71e-15 | 86.27% |
| **TAL** | **51** | **5.23e-22** | **96.08%** |

## Struttura

- `01_abstract.md` — abstract
- `02_introduction.md` — introduzione
- `03_related_work.md` — stato dell'arte
- `04_method.md` — descrizione di TAL
- `05_experimental_setup.md` — setup
- `06_results.md` — risultati
- `07_discussion.md` — discussione
- `08_conclusion.md` — conclusioni
- `09_references.md` — bibliografia
- `10_statistical_tests.md` - analisi statistica
- `data/` — dati del benchmark
- `figures/` — figure pubblicabili
- `scripts/` — script di analisi
