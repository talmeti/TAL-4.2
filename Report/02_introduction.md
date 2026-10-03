# 1. Introduzione

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
