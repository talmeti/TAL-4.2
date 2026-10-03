# 6. Discussione

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
