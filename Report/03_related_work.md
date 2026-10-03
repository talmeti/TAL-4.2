# 2. Stato dell'arte

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
