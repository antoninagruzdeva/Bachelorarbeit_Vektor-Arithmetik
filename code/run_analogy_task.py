import os
import json
from vecto.embeddings import load_from_dir
from vecto.benchmarks.analogy.analogy import Analogy
from vecto.data import Dataset


DATASET_DIR = "BATS_shared"

def run_analogy_for_model(model_dir: str, dataset_dir: str, method: str):
    """
    Führt den Analogie-Task für ein eingebettetes Modell mit einer einzigen Methode aus
    und speichert die Ergebnisse im entsprechenden Ergebnisverzeichnis.
    """
    # 1. Modell laden
    model = load_from_dir(model_dir)
    # 2. Datensatz initialisieren
    dataset = Dataset(dataset_dir)

    # 3. Ableiten von Modell- und Methodennamen
    model_name = os.path.basename(model_dir)
    # Ausgabeordner: analogy_results_<model_name>
    out_dir = f"analogy_results_{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    # 4. Benchmark-Setup
    analogy_task = Analogy(method=method, normalize=True, exclude=True)
    # 5. Ausführen
    results = analogy_task.run(model, dataset)

    # 6. Dateiname: <model_name>_<method>.json
    fname = f"{model_name}_{method}.json"
    output_path = os.path.join(out_dir, fname)

    # 7. Speichern
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Ergebnisse gespeichert unter: {output_path}")


if __name__ == '__main__':
    model_dir = "pythia_cleaned_embs"
    method    = "3CosAdd"               ## OnlyB heißt in der Vecto-Library SimilarToB
    run_analogy_for_model(model_dir, DATASET_DIR, method)
