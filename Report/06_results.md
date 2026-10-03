# 5. Risultati

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

## 5.6 Figure

### Figura 1: boxplot per metodo

![Figura 1](figures/fig1_boxplot_method.png)

### Figura 2: solved rate

![Figura 2](figures/fig2_solved_rate.png)

### Figura 3: boxplot per dominio

![Figura 3](figures/fig3_boxplot_domain.png)

### Figura 4: trade-off tempo/errore

![Figura 4](figures/fig4_scatter_time_error.png)
