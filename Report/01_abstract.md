# Abstract

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
