# TAL - Theory-Adaptive Learning

**TAL 4.2.0** - Compositional Symbolic Regression with Adaptive Grammar Selection.

TAL e' un framework di **regressione simbolica** che scopre espressioni
in forma chiusa a partire da dati, combinando:

- **due grammatiche** (base: 8 operatori; estesa: 13 operatori),
- **selezione adattiva** della grammatica migliore per ogni funzione,
- **fitter multi-start** basato su lmfit,
- **memoria parametrica** che riusa i parametri tra alberi con la stessa firma,
- **criterio di accettazione deterministico** basato sul miglioramento dell'MSE.

## Risultati

Su 17 funzioni benchmark distribuite in 4 domini:

| Dominio   | Funzioni | Risolte | Perfette | Base | Estesa |
|-----------|---------:|--------:|---------:|-----:|-------:|
| Feynman   |        8 |       8 |        8 |    4 |      4 |
| ODE       |        3 |       3 |        3 |    2 |      1 |
| Dinamici  |        3 |       2 |        2 |    2 |      1 |
| Reali     |        3 |       3 |        2 |    2 |      1 |
| **Totale**|   **17** |  **16** |   **15** |**10**|  **7** |

- Risolta: MSE di test < 0.01
- Perfetta: MSE di test < 0.001

## Installazione

### Con conda (consigliato)

    conda env create -f environment.yml
    conda activate tal

### Con pip

    pip install -r requirements.txt

## Uso

    python tal_4_2_step01_cleanup.py

Output:

- tal_4_2_output.txt - log completo dell'esecuzione
- tal_4_2.json - risultati in formato JSON
- tal_4_2_step01.log - log dettagliato (DEBUG)

## Test

    pytest tal_4_2_step02_tests.py -v

41 test che coprono grammatiche, utility, memoria, fitter, generator,
domini e due casi end-to-end.

## Struttura del progetto

    .
    tal_4_2_step01_cleanup.py         # pipeline principale
    tal_4_2_step02_tests.py           # suite pytest
    tal_4_2_step03_reproducibility.py # report d'ambiente
    requirements.txt
    environment.yml
    pyproject.toml
    CITATION.cff
    LICENSE
    README.md

## Roadmap

- [x] Step 01 - cleanup + riproducibilita verificata
- [x] Step 02 - 41 test pytest
- [x] Step 03 - packaging, licenza, citazione
- [ ] Step 04 - benchmark multi-seed, rumore, extrapolazione, baseline
- [ ] Step 05 - ablation (memoria, grammatica estesa, detector)
- [ ] Step 06 - paper / release pubblica

## Citazione

    @software{fenech_casoni_tal_2026,
      author       = {Fenech-Casoni},
      title        = {{TAL - Theory-Adaptive Learning}},
      version      = {4.2.0},
      year         = {2026},
      license      = {MIT},
      url          = {https://github.com/fenech-casoni/tal}
    }

## Licenza

MIT - vedi LICENSE.
