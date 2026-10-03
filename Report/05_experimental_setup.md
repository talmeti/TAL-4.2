# 4. Setup sperimentale

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
