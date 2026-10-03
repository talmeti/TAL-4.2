# 3. Metodo: TAL 4.2

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
