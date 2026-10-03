# TAL 4.2 — Theory-Adaptive Learning

**Symbolic regression with adaptive grammars.**

**Autori**: Ruggero Fenech, Giorgio Casoni

---

## Panoramica

TAL 4.2 (Theory-Adaptive Learning) è un metodo di **symbolic regression**
che combina grammatiche adattive, memoria parametrica e un criterio di
accettazione deterministico per guidare la ricerca.

Il metodo è confrontato con:
- **poly**: baseline polinomiale (grado 5)
- **PySR**: stato dell'arte nella symbolic regression

Su 17 problemi in 4 domini (Feynman, ODE, Dinamici, Reali), per un
totale di 153 run.

## Risultati principali

| Metodo | n | MSE median | Solved rate |
|--------|---|------------|-------------|
| poly | 51 | 4.32e-04 | 60.78% |
| PySR | 51 (44 ok) | 1.71e-15 | 86.27% |
| **TAL** | **51** | **5.23e-22** | **96.08%** |

- TAL è **significativamente migliore di poly** su tutti i domini (p < 1e-5).
- TAL è **significativamente migliore di PySR su Feynman** (p = 0.012).
- TAL è **statisticamente equivalente a PySR** su ODE, Dinamici e Reali.
- TAL è **più stabile** (0 fallimenti su 51 run vs 7 di PySR).
- TAL è **~4x più lento** di PySR (~60 s vs ~14 s per run).

## Report completo

Il report completo è disponibile in tre formati:

- **[Report/README.md](Report/README.md)** — Report in formato Markdown (sorgente)
- **[Report/Report.pdf](Report/Report.pdf)** — Report in formato PDF (leggibile, stampabile)
- **[Report/report.html](Report/report.html)** — Report in formato HTML (interattivo)

Il report include:
- Abstract e introduzione
- Stato dell'arte (PySR, AI Feynman)
- Descrizione del metodo TAL 4.2
- Setup sperimentale
- Risultati (globali e per dominio)
- Discussione e conclusioni
- Analisi statistica completa (Mann-Whitney U, Wilcoxon, Cohen's d)

## Codice

- **[tal_4_2_step01_cleanup.py](tal_4_2_step01_cleanup.py)** — Modulo principale (grammatiche, fitter, generator)
- **[tal_4_2_step04_benchmark.py](tal_4_2_step04_benchmark.py)** — Benchmark (TAL vs poly vs PySR)

## Riproducibilità

### Requisiti

```bash
pip install -r requirements.txt
```

### Eseguire il benchmark

```bash
python tal_4_2_step04_benchmark.py --quick --with-pysr --n-jobs 8
```

### Generare le figure del report

```bash
cd Report/scripts
python make_paper_figures.py
python analyze_by_domain.py
python full_statistical_analysis.py
```

### Convertire il report in HTML/PDF

```bash
cd Report
pandoc README.md 01_abstract.md 02_introduction.md 03_related_work.md 04_method.md 05_experimental_setup.md 06_results.md 07_discussion.md 08_conclusion.md 09_references.md 10_statistical_tests.md -o report.html --embed-resources --standalone
```

## Citazione

Se usi questo lavoro, cita:

```bibtex
@techreport{fenech2026tal,
  title = {{TAL 4.2: Theory-Adaptive Learning for Symbolic Regression}},
  author = {Fenech, Ruggero and Casoni, Giorgio},
  year = {2026},
  month = {10},
  type = {Technical Report},
  url = {https://github.com/talmeti/TAL-4.2}
}
```

## Licenza

Vedi il file [LICENSE](LICENSE) per i dettagli.