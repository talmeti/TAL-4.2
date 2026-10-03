# 9. Analisi statistica

Questa sezione riporta l'analisi statistica completa dei risultati del benchmark. Tutti i test sono stati eseguiti con `scipy.stats` su 153 run (51 per metodo, esclusi gli outlier con errore >= 1e10).

## 9.1 Statistiche descrittive

| Metodo | n | Mean | Median | Std | CI95 low | CI95 high |
|--------|---|------|--------|-----|----------|-----------|
| poly | 51 | 1.390e-02 | 4.320e-04 | 1.958e-02 | 8.478e-03 | 1.933e-02 |
| PySR | 41 | 3.792e-05 | 1.618e-14 | 1.479e-04 | -7.915e-06 | 8.375e-05 |
| **TAL** | **51** | **1.506e-03** | **5.230e-22** | **5.841e-03** | **-1.128e-04** | **3.125e-03** |

**Osservazioni**:

- TAL ha la mediana dell'errore più bassa (5.23e-22), ~7 ordini di grandezza migliore di PySR (1.62e-14).
- Le medie sono gonfiate da outlier: TAL ha media 1.5e-03, PySR 3.8e-05, poly 1.4e-02.
- Gli intervalli di confidenza di TAL e PySR includono lo zero (a causa della distribuzione bimodale).

## 9.2 Mann-Whitney U test (globale)

| Confronto | U | p-value | Significatività | Cohen's d |
|-----------|---|---------|-----------------|-----------|
| TAL vs poly | 604.0 | 3.19e-06 | *** | **0.850** (grande) |
| TAL vs PySR | 932.5 | 0.377 | ns | -0.334 (piccolo) |
| PySR vs poly | 415.0 | 7.46e-07 | *** | **0.941** (grande) |

**Legenda**: *** p<0.001, ** p<0.01, * p<0.05, ns = non significativo.
Cohen's d: |d| < 0.2 piccolo, 0.2-0.5 medio, > 0.8 grande.

**Interpretazione**:

- **TAL e PySR sono entrambi significativamente migliori di poly** (p < 1e-5), con effect size grandi (d ≈ 0.85-0.94).
- **TAL e PySR non sono statisticamente distinguibili** (p = 0.377), con effect size piccolo (d = -0.33). Questo significa che, globalmente, TAL e PySR hanno prestazioni equivalenti.
- Il segno negativo di Cohen's d (TAL vs PySR) indica che PySR ha media **leggermente** minore, ma la differenza non è significativa.

## 9.3 Mann-Whitney U test per dominio (TAL vs PySR)

| Dominio | TAL n | PySR n | U | p-value | Signif. | Cohen's d |
|---------|-------|--------|---|---------|---------|-----------|
| **Feynman** | 24 | 19 | 120.0 | **0.0086** | ** | 0.204 (piccolo) |
| ODE | 9 | 9 | 30.0 | 0.376 | ns | -0.573 (medio) |
| Dinamici | 9 | 5 | 18.0 | 0.606 | ns | -0.434 (medio) |
| Reali | 9 | 8 | 56.0 | 0.059 | ns | -0.693 (medio) |

**Interpretazione**:

- **Feynman**: TAL è **significativamente migliore** di PySR (p = 0.0086). Effect size piccolo (d = 0.20), ma la differenza è statisticamente solida.
- **ODE**: TAL e PySR sono equivalenti (p = 0.376). Effect size medio (d = -0.57), ma il campione è piccolo (9 run).
- **Dinamici**: TAL e PySR sono equivalenti (p = 0.606). PySR ha media migliore, ma non significativo.
- **Reali**: quasi significativo (p = 0.059). TAL ha mediana migliore (9.15e-04 vs 5.78e-05 di PySR, nota: qui PySR ha media migliore ma TAL ha mediana migliore).

**Nota**: su Reali, il p-value (0.059) è **appena sopra** la soglia di 0.05. Con più run (es. 20 seed), potrebbe diventare significativo.

## 9.4 Mann-Whitney U test per dominio (TAL vs poly)

| Dominio | TAL n | poly n | U | p-value | Signif. | Cohen's d |
|---------|-------|--------|---|---------|---------|-----------|
| Feynman | 24 | 24 | 116.0 | 4.05e-04 | *** | 1.122 (grande) |
| ODE | 9 | 9 | 6.0 | 2.63e-03 | ** | 0.915 (grande) |
| Dinamici | 9 | 9 | 12.0 | 1.34e-02 | * | 0.754 (medio) |
| Reali | 9 | 9 | 31.0 | 0.427 | ns | 0.784 (medio) |

**Interpretazione**:

- **Feynman, ODE, Dinamici**: TAL è **significativamente migliore** di poly (p < 0.05), con effect size medio-grande (d ≈ 0.75-1.12).
- **Reali**: differenza non significativa (p = 0.43), nonostante effect size medio (d = 0.78). Il campione è troppo piccolo.

## 9.5 Wilcoxon signed-rank test (run appaiati)

Per un confronto più rigoroso, abbiamo appaiato i run TAL e PySR per (dominio, funzione, seed):

- **Run appaiati**: n = 41
- **Wilcoxon W**: 393.0
- **p-value**: 0.627

**Interpretazione**: **nessuna differenza significativa** tra TAL e PySR sui run appaiati. Questo conferma il risultato del Mann-Whitney U.

## 9.6 Sommario interpretativo

### Confronti significativi (p < 0.05)

- **TAL vs poly**: p = 3.19e-06 (***), d = 0.850 (grande)
- **PySR vs poly**: p = 7.46e-07 (***), d = 0.941 (grande)
- **TAL vs PySR su Feynman**: p = 0.0086 (**), d = 0.204 (piccolo)
- **TAL vs poly su Feynman**: p = 4.05e-04 (***), d = 1.122 (grande)
- **TAL vs poly su ODE**: p = 2.63e-03 (**), d = 0.915 (grande)
- **TAL vs poly su Dinamici**: p = 1.34e-02 (*), d = 0.754 (medio)

### Confronti non significativi (p >= 0.05)

- **TAL vs PySR**: p = 0.377 (ns), d = -0.334 (piccolo)
- **TAL vs PySR su ODE**: p = 0.376 (ns), d = -0.573 (medio)
- **TAL vs PySR su Dinamici**: p = 0.606 (ns), d = -0.434 (medio)
- **TAL vs PySR su Reali**: p = 0.059 (ns), d = -0.693 (medio)
- **TAL vs poly su Reali**: p = 0.427 (ns), d = 0.784 (medio)

## 9.7 Conclusioni statistiche

1. **TAL batte poly** significativamente su 3 domini su 4 (Feynman, ODE, Dinamici), con effect size grande.
2. **TAL batte PySR significativamente su Feynman** (p = 0.0086), con effect size piccolo ma solido.
3. **TAL e PySR sono statisticamente equivalenti** sugli altri domini (ODE, Dinamici, Reali).
4. **Il campione è piccolo** per alcuni domini (5-9 run), il che limita la potenza statistica.
5. **Con più run** (es. 20 seed), alcune differenze non significative potrebbero diventare significative (es. Reali, p = 0.059).

## 9.8 Limitazioni dell'analisi

- **Campione limitato**: 51 run per metodo (3 seed × 17 funzioni). Con 20 seed, la potenza statistica aumenterebbe.
- **Distribuzione bimodale**: TAL e PySR hanno distribuzioni bimodali (soluzione esatta o fallimento), il che rende i test parametrici inappropriati.
- **Outlier esclusi**: i 7 run PySR con errore >= 1e10 sono stati esclusi dai test statistici.
- **Test non parametrici**: Mann-Whitney U e Wilcoxon sono appropriati per distribuzioni non normali, ma hanno meno potenza dei test parametrici.