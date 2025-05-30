import os
import json
from vecto.embeddings import load_from_dir
from vecto.benchmarks.analogy.analogy import Analogy
from vecto.data import Dataset
import re


DATASET_DIR = "BATS_shared"

def run_analogy_for_model(model_dir: str, dataset_dir: str, method: str):
    """
    Führt den Analogie-Task für ein eingebettetes Modell mit einer einzigen Methode aus
    und speichert die Ergebnisse im entsprechenden Ergebnisverzeichnis.
    """

    model = load_from_dir(model_dir)
    dataset = Dataset(dataset_dir)
    model_name = os.path.basename(model_dir)
    base_model = re.sub(r'_(embs|unembs)$', '', model_name)
    out_dir = f"analogy_results_{base_model}_test"
    os.makedirs(out_dir, exist_ok=True)
    analogy_task = Analogy(method=method, normalize=True, exclude=True)
    results = analogy_task.run(model, dataset)
    fname = f"{model_name}_{method}.json"
    output_path = os.path.join(out_dir, fname)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Ergebnisse gespeichert unter: {output_path}")


if __name__ == '__main__':
    model_dir = "pythia_cleaned_embs"
    method    = "3CosAdd"               ## OnlyB heißt in der Vecto-Library SimilarToB!
    run_analogy_for_model(model_dir, DATASET_DIR, method)
