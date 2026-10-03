# 7. Conclusioni

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
