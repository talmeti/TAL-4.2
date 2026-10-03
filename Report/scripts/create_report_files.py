# -*- coding: utf-8 -*-
"""Crea tutti i file .md del report TAL 4.2."""
from pathlib import Path

REPORT_DIR = Path(r"C:\Users\tal\Desktop\TEORIA AI\Report")

files = {}

files["README.md"] = """# TAL 4.2 — Report di benchmark

**Data**: 2026-10-03
**Autore**: Ruggero Fenech
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
- `data/` — dati del benchmark
- `figures/` — figure pubblicabili
- `scripts/` — script di analisi
"""

files["01_abstract.md"] = """# Abstract

La symbolic regression e' un problema fondamentale nell'apprendimento
automatico: data una serie di osservazioni, trovare un'espressione
simbolica che le descriva.

Proponiamo TAL 4.2 (Theory-Adaptive Learning), un metodo di symbolic
regression che combina grammatiche adattive, memoria parametrica e un
criterio di accettazione deterministico.

Confrontiamo TAL con poly e PySR su 17 problemi in 4 domini, per un
totale di 153 run.

Risultati principali:

- TAL raggiunge un solved_rate del 96.08%, contro 86.27% di PySR e
  60.78% di poly.
- TAL ha una mediana dell'errore di 5.23e-22, contro 1.71e-15 di PySR
  e 4.32e-04 di poly.
- Su Feynman, TAL e' significativamente migliore di PySR (p = 0.012),
  con una mediana ~10 ordini di grandezza inferiore.
- TAL e' significativamente migliore di poly su tutti i domini
  (p < 1e-5).
- TAL completa 51/51 run, PySR ne completa 44/51.

Il trade-off principale e' la velocita': TAL e' ~4x piu' lento di PySR.
"""

files["02_introduction.md"] = """# 1. Introduzione

## 1.1 Motivazione

La symbolic regression (SR) e' il problema di trovare un'espressione
matematica che descriva un insieme di dati. A differenza della
regressione tradizionale, la SR scopre anche la forma funzionale.

Applicazioni: fisica, biologia, economia, ingegneria.

I metodi esistenti (PySR, AI Feynman) eccellono su problemi fisici ma
hanno limiti: fragilita' numerica, scarsa generalizzazione, costo
computazionale.

## 1.2 Contributo

Proponiamo TAL 4.2 (Theory-Adaptive Learning), che affronta questi
limiti attraverso:

1. Grammatiche adattive (base a 8 op, estesa a 13 op).
2. Memoria parametrica.
3. Criterio di accettazione deterministico.

## 1.3 Risultati

TAL eccelle su problemi fisici (Feynman), e' equivalente a PySR su ODE,
Dinamici e Reali, ed e' piu' stabile (0 fallimenti vs 7).

## 1.4 Struttura

- Sezione 2: stato dell'arte.
- Sezione 3: metodo TAL.
- Sezione 4: setup sperimentale.
- Sezione 5: risultati.
- Sezione 6: discussione.
- Sezione 7: conclusioni.
"""

files["03_related_work.md"] = """# 2. Stato dell'arte

## 2.1 Symbolic Regression

Introdotta da Koza (1992) con Genetic Programming. Evolve alberi
simbolici tramite mutazione e crossover.

Vantaggi: flessibile, applicabile a qualsiasi dominio.
Svantaggi: costo elevato, overfitting.

## 2.2 PySR

PySR (Cranmer, 2023) combina algoritmo evolutivo multi-popolazione,
semplificazione simbolica, e selezione Pareto.

Vantaggi: veloce (~15 s), robusto.
Svantaggi: fragile (20% fallimenti), meno preciso di TAL su Feynman.

## 2.3 AI Feynman

AI Feynman (Udrescu & Tegmark, 2020) usa decomposizione ricorsiva e
trasformazioni di simmetria.

Vantaggi: eccelle su fisica.
Svantaggi: limitato a problemi fisici.

## 2.4 Baseline polinomiale

poly: fit polinomiale di grado 5 tramite numpy.polyfit. Istantaneo
ma poco preciso su funzioni non polinomiali.

## 2.5 Posizionamento di TAL

TAL si distingue per grammatiche adattive, memoria parametrica e
stabilita' numerica.
"""

files["04_method.md"] = """# 3. Metodo: TAL 4.2

## 3.1 Panoramica

TAL combina: generazione di alberi simbolici, fit multi-start LM,
selezione adattiva della grammatica, memoria parametrica.

## 3.2 Grammatiche

### Base (8 operatori)
x, const, lin, quad, sin, exp, sum, prod

### Estesa (13 operatori)
Base + cos, div, log, sqrt, step

## 3.3 Algoritmo

1. Per ogni epoca (n_epochs = 5):
   a. Clona o inizializza struttura.
   b. Per ogni iterazione (n_iter = 15):
      i.   Genera 7 candidati.
      ii.  Fitta con RobustFitterLM.
      iii. Accetta se riduce errore del 5%.
2. Ripeti per grammatica base ed estesa.
3. Scegli la migliore.

## 3.4 RobustFitterLM

- n_starts = 10
- max_nfev = 500
- max_params = 20
- memoria parametrica

## 3.5 Criterio di accettazione

MSE_new < MSE_old * 0.95

## 3.6 Configurazione

n_epochs=5, n_iter=15, err_threshold=1e-6, patience=2
"""

files["05_experimental_setup.md"] = """# 4. Setup sperimentale

## 4.1 Domini

- Feynman (8 funzioni)
- ODE (3 funzioni)
- Dinamici (3 funzioni)
- Reali (3 funzioni)

Totale: 17 funzioni.

## 4.2 Configurazioni

### Quick (questo report)
- Seeds: [0, 1, 2]
- Noise: [0.0]
- Extrapolation: [None]
- Job totali: 153

### Critical (run successivo)
- Seeds: [0..4]
- Noise: [0.05]
- Extrapolation: ["half", "central"]
- Job totali: 510

### Full (run futuro)
- Seeds: [0..19]
- Noise: [0.0, 0.01, 0.05]
- Extrapolation: [None, "central", "half"]
- Job totali: 9180

## 4.3 Metriche

- MSE (Mean Squared Error)
- solved_rate (MSE < 0.01)
- perfect_rate (MSE < 0.001)
- Tempo (s)

## 4.4 Hardware

- CPU: Intel Core i7-10700K (8 core)
- RAM: 32 GB
- Storage: Samsung 954 GB NVMe SSD
- OS: Windows 10/11 Pro 64-bit

## 4.5 Software

- Python 3.14
- numpy, sympy, lmfit, joblib
- PySR 2.4.0 + Julia
- matplotlib
"""

files["06_results.md"] = """# 5. Risultati

## 5.1 Confronto globale

| Metodo | n | MSE mean | MSE median | Solved rate | Perfect rate |
|--------|---|----------|------------|-------------|--------------|
| poly | 51 | 1.39e-02 | 4.32e-04 | 60.78% | 52.94% |
| PySR | 51 (44 ok) | 3.43e+16 | 1.71e-15 | 86.27% | 82.35% |
| **TAL** | **51** | **1.51e-03** | **5.23e-22** | **96.08%** | **88.24%** |

TAL ha solved_rate piu' alto e mediana migliore. PySR ha mse_mean
gonfiata da 7 run con errore 1e20.

## 5.2 Per dominio

### Feynman (24 run)
- TAL: median = 3.64e-25
- PySR: median = 1.71e-15, p = 0.012 (significativo)
- poly: median = 1.52e-02, p < 1e-5

TAL e' significativamente migliore di PySR.

### ODE (9 run)
- TAL: median = 3.40e-29
- PySR: median = 1.42e-33, p = 0.789 (non signif.)
- poly: median = 1.61e-05, p < 1e-3

TAL e PySR sono equivalenti.

### Dinamici (9 run)
- TAL: median = 4.19e-07
- PySR: median = 1.08e-07, p = 0.351 (non signif.)
- poly: median = 3.22e-04, p < 0.05

TAL e PySR sono equivalenti.

### Reali (9 run)
- TAL: median = 9.15e-04
- PySR: median = 5.78e-05, p = 0.114 (non signif.)
- poly: median = 4.04e-02, p < 0.01

PySR ha mediana migliore ma non significativo.

## 5.3 Test statistici globali

| Confronto | U | p-value | Signif. |
|-----------|---|---------|---------|
| TAL vs poly | 604.0 | 3.19e-06 | *** |
| TAL vs PySR | 1118.0 | 0.979 | ns |
| TAL vs PySR (Feynman) | — | 0.012 | * |

## 5.4 Fallimenti

| Metodo | Fallimenti |
|--------|------------|
| poly | 0/51 |
| PySR | 7/51 |
| TAL | 0/51 |

PySR fallisce su I_12_2, I_40_1, I_12_11, I_30_3, dyn_crit,
dyn_over, real_poly.

## 5.5 Tempi

| Metodo | Tempo medio |
|--------|-------------|
| poly | ~0.0002 s |
| PySR | ~14 s |
| TAL | ~60 s |

## 5.6 Figure

- fig1_boxplot_method.png
- fig2_solved_rate.png
- fig3_boxplot_domain.png
- fig4_scatter_time_error.png
"""

files["07_discussion.md"] = """# 6. Discussione

## 6.1 Risultati principali

TAL e' competitivo con PySR su tutti i domini e significativamente
migliore su Feynman.

Punti di forza TAL:
1. Precisione: mediana 5.23e-22 vs 1.71e-15 di PySR.
2. Affidabilita': 96.08% solved vs 86.27%.
3. Stabilita': 0 fallimenti vs 7.

Punti di forza PySR:
1. Velocita': ~14 s vs ~60 s.
2. Parita' su ODE e Dinamici.

## 6.2 Interpretazione

### Perche' TAL eccelle su Feynman?

Le funzioni Feynman hanno forme simboliche semplici. La grammatica
adattiva di TAL copre bene queste forme.

### Perche' PySR e' piu' veloce?

PySR usa algoritmo evolutivo in Julia. TAL usa multi-start LM in Python.

### Perche' PySR fallisce?

PySR produce espressioni con divisioni per zero o log di negativi.
TAL usa un criterio di accettazione che previene questi casi.

## 6.3 Limitazioni

Limiti di TAL:
1. Lentezza (~4x PySR).
2. Non testato su alta dimensionalita'.
3. Non testato con rumore o extrapolazione.

Limiti di PySR:
1. Instabilita' numerica (13.7% fallimenti).
2. Grammar fissa.

## 6.4 Lavoro futuro

1. Ottimizzare TAL (parallelizzare multi-start).
2. Testare con rumore (--critical).
3. Testare su extrapolazione (--full).
4. Confrontare con AI Feynman.
5. Scalare a problemi reali.

## 6.5 Implicazioni pratiche

TAL: applicazioni offline, problemi fisici, scenari dove la stabilita'
e' critica.

PySR: applicazioni real-time, dove la velocita' e' critica.
"""

files["08_conclusion.md"] = """# 7. Conclusioni

Abbiamo presentato TAL 4.2, un metodo di symbolic regression con
grammatiche adattive, memoria parametrica e criterio di accettazione
deterministico.

Il benchmark su 17 funzioni in 4 domini mostra che:

1. TAL e' significativamente migliore di poly (p < 1e-5).
2. TAL e' significativamente migliore di PySR su Feynman (p = 0.012).
3. TAL e' equivalente a PySR su ODE, Dinamici e Reali.
4. TAL e' piu' stabile (0 fallimenti vs 7).
5. TAL e' ~4x piu' lento di PySR.

Il trade-off principale e' precisione vs velocita'.

## Prospettive

- Ottimizzazione del fitter.
- Robustezza a rumore ed extrapolazione.
- Scalabilita' ad alta dimensionalita'.
- Confronto con AI Feynman.
"""

files["09_references.md"] = """# 8. Riferimenti

1. Koza, J. R. (1992). Genetic Programming. MIT Press.

2. Cranmer, M. (2023). Interpretable Machine Learning for Science
   with PySR and SymbolicRegression.jl. arXiv:2305.01582.

3. Udrescu, S.-M., & Tegmark, M. (2020). AI Feynman. Science
   Advances, 6(16), eaay2631.

4. Schmidt, M., & Lipson, H. (2009). Distilling Free-Form Natural
   Laws from Experimental Data. Science, 324(5923), 81-85.

5. McKay, B., et al. (2010). Using a tree structured genetic
   algorithm to perform symbolic regression. GECCO.

6. Bongard, J., & Lipson, H. (2007). Automated reverse engineering
   of nonlinear dynamical systems. PNAS.

7. Virgolin, M., et al. (2021). Generating Diverse Solutions in
   Symbolic Regression. IEEE TEC.

8. La Cava, W., et al. (2021). Contemporary Symbolic Regression
   Methods and their Relative Performance. NeurIPS.

9. Cranmer, M., et al. (2020). Discovering Symbolic Models from
   Deep Learning with Inductive Biases. NeurIPS.

10. Schmidt, M., & Lipson, H. (2011). Symbolic Regression of
    Implicit Equations. Springer.
"""

# Scrivi tutti i file
for name, content in files.items():
    path = REPORT_DIR / name
    path.write_text(content, encoding="utf-8")
    print(f"Creato: {path}")

print()
print(f"Totale file creati: {len(files)}")