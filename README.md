# Bachelorarbeit: Untersuchung von Embedding-Arithmetik in Transformer-Modellen

Dieses Repository enthält den vollständigen Code und die begleitenden Daten zur Bachelorarbeit *„Untersuchung von Embedding-Arithmetik in Transformer-Modellen“*. Ziel der Arbeit war es, die Fähigkeit klassischer Vektorraum-Modelle (Word2Vec) und moderner Transformer-Modelle (BERT, GPT-2, Pythia 410M) zur Abbildung linguistischer Relationen mithilfe etablierter Analogie-Metriken zu evaluieren.

## Inhalte des Repositories

```
.
├── code/                        # Python-Skripte zur Vorbereitung, Analyse und Auswertung
│   ├── extract_shared_tokens.py
│   ├── extract_submatrix_transformer.py
│   ├── extract_submatrix_word2vec.py
│   ├── filter_bats_to_shared.py
│   ├── run_analogy_task.py
│   ├── shared_token_utils.py
│   └── summarize_and_plot_json_results.py
├── data/                        # Datensätze und Ergebnisse
│   ├── BATS shared/             # Gefilterter BATS 3.0 Analogie-Datensatz
│   ├── results_json/           # JSON-Ausgaben der Analogieexperimente
│   ├── vectors filtered/       # Modellsubmatrizen (Embedding / Unembedding)
│   ├── auswertung_top1_accuracy_similarity.csv
│   ├── auswertung_top5_accuracy_similarity.csv
│   └── shared tokens.txt       # Liste der gemeinsamen Tokens (14.181)
├── thesis.pdf                  # Vollständige Bachelorarbeit (PDF)
└── README.md                   
```

## Experimentelle Vorgehensweise

- **Datengrundlage**: BATS 3.0, gefiltert auf ein gemeinsames Vokabular aller Modelle (14.181 Tokens)
- **Modelle**: Word2Vec, BERT (base uncased), GPT-2 (small), Pythia 410M
- **Analogie-Metriken**: 3CosAdd, 3CosAvg, PairDistance, OnlyB(Baseline)

## Setup

Dieses Projekt wurde mit **Python 3.12.0** entwickelt. Zum Ausführen der Skripte sollten folgende Libraries installiert sein:

- `numpy`
- `pandas`
- `matplotlib`
- `glob`
- `transformerlens`
- `gensim`
- `vecto`
- `os`
- `re`
- `json`

## Nutzung

Zur Reproduktion der Experimente wird zunächst der **BATS 3.0 Datensatz** benötigt, der [hier](https://u.pcloud.link/publink/show?code=XZOn0J7Z8fzFMt7Tw1mGS6uI1SYfCfTyJQTV) öffentlich verfügbar ist. Dieser muss lokal gespeichert werden, z. B. im Ordner `data/BATS_3.0`.

Anschließend erfolgt die Durchführung des Experiments in den folgenden Schritten:

1. **Modelle laden, Vokabulare normalisieren und gemeinsame Tokens extrahieren**  
   → `extract_shared_tokens.py`

2. **Embedding- und Unembedding-Matrizen extrahieren und auf das gemeinsame Vokabular filtern**  
   → `extract_submatrix_transformer.py` (für GPT-2, BERT, Pythia)  
   → `extract_submatrix_word2vec.py` (für Word2Vec)

3. **BATS-Datensatz auf das gemeinsame Vokabular filtern**  
   → `filter_bats_to_shared.py`

4. **Analogie-Aufgaben auf den vorbereiteten Matrizen ausführen**  
   → `run_analogy_task.py`

5. **Ergebnisse auswerten und visualisieren**  
   → `summarize_and_plot_json_results.py`


## Lizenz

Dieses Projekt dient ausschließlich wissenschaftlichen und nicht-kommerziellen Zwecken im Rahmen einer Abschlussarbeit.

## Kontakt

Für Rückfragen oder Feedback zur Arbeit:  
**Antonina Gruzdeva** – Antonina.Gruzdeva@campus.lmu.de

