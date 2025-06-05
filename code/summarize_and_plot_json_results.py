
import matplotlib
matplotlib.use('TkAgg')
import os
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

COLOR_MAP = {
    "OnlyB": "#1f77b4",
    "3CosAvg": "#ff7f0e",
    "3CosAdd": "#2ca02c",
    "PairDistance": "#d62728"
}

def load_mean_accuracy_and_similarity_per_subcategory(root_dirs: list) -> pd.DataFrame:
    """
    Lädt Accuracy (aus result.accuracy) und Top-1 Similarity aus allen JSON-Dateien
    mit dem Format <model_name>_<method>.json.

    Gibt DataFrame mit Spalten:
    ['model', 'method', 'category', 'subcategory', 'accuracy', 'similarity']
    """
    records = []
    for root in root_dirs:
        pattern = os.path.join(root, '*.json')
        for json_file in glob.glob(pattern):
            fname = os.path.basename(json_file)
            base = os.path.splitext(fname)[0]
            parts = base.split('_')
            method = parts[-1]
            model_name = '_'.join(parts[:-1])
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for entry in data:
                cat = entry.get('experiment_setup', {}).get('category', 'Unknown')
                sub = entry.get('experiment_setup', {}).get('subcategory', 'Unknown')
                acc = entry.get('result', {}).get('accuracy', None)

                sim_vals = []
                for det in entry.get('details', []):
                    preds = det.get('predictions', [])
                    if preds:
                        top_pred = preds[0]
                        if top_pred.get('hit', False):
                            score = top_pred.get('score')
                            sim_vals.append(score)

                top_sim = sum(sim_vals) / len(sim_vals) if sim_vals else None

                records.append({
                    'model': model_name,
                    'method': method,
                    'category': cat,
                    'subcategory': sub,
                    'accuracy': acc,
                    'similarity': top_sim
                })

    df1 = pd.DataFrame(records)
    print(f"Geladen: {len(df1)} Subkategorie-Einträge aus {len(root_dirs)} Ordnern")
    df1.to_csv("auswertung_top1_accuracy_similarity.csv", index=False)
    return df1

def load_top5_accuracy_and_similarity_per_subcategory(root_dirs: list) -> pd.DataFrame:
    """
    Lädt Top-5 Accuracy und mittlere Similarity
    über alle Treffer innerhalb der Top-5 aus allen JSON-Dateien.

    Gibt DataFrame mit Spalten:
    ['model', 'method', 'category', 'subcategory', 'accuracy', 'similarity']
    """
    records = []
    for root in root_dirs:
        pattern = os.path.join(root, '*.json')
        for json_file in glob.glob(pattern):
            fname = os.path.basename(json_file)
            base = os.path.splitext(fname)[0]
            parts = base.split('_')
            method = parts[-1]
            model_name = '_'.join(parts[:-1])

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for entry in data:
                cat = entry.get('experiment_setup', {}).get('category')
                sub = entry.get('experiment_setup', {}).get('subcategory')

                hit_count = 0
                sim_vals = []

                for det in entry.get('details', []):
                    preds = det.get('predictions', [])
                    found = False
                    for pred in preds[:5]:  # Nur Top-5 berücksichtigen
                        if pred.get('hit', False):
                            hit_count += 1
                            score = pred.get('score')
                            if score is not None:
                                sim_vals.append(score)
                            found = True
                            break

                total = len(entry.get('details', []))
                acc_top5 = hit_count / total if total > 0 else None
                sim_top5 = sum(sim_vals) / len(sim_vals) if sim_vals else None

                records.append({
                    'model': model_name,
                    'method': method,
                    'category': cat,
                    'subcategory': sub,
                    'accuracy': acc_top5,
                    'similarity': sim_top5
                })

    df5 = pd.DataFrame(records)
    print(f"Geladen: {len(df5)} Subkategorie-Einträge aus {len(root_dirs)} Ordnern")
    df5.to_csv("auswertung_top5_accuracy_similarity.csv", index=False)
    return df5

def summarize_accuracy_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gibt mittlere Accuracy pro Modell, Methode und Hauptkategorie zurück.
    """
    summary = (
        df.groupby(['model', 'method', 'category'])
        .agg(mean_accuracy=('accuracy', 'mean'))
        .reset_index()
        .round(3)
    )
    print("\n Mittlere Accuracy pro Modell / Methode / Kategorie:")
    print(summary.to_string(index=False))
    return summary

def plot_accuracy_by_category(df: pd.DataFrame, model_name: str, out_dir: str = 'plots'):
    """
    Für ein Modell wird die Accuracy pro Kategorie dargestellt,
    gruppiert nach Methode (OnlyB, 3CosAvg, ...), mit Farben aus COLOR_MAP.
    """
    df_model = df[df['model'] == model_name]

    # Pivot: Kategorien = Zeilen, Methoden = Spalten, Werte = Accuracy
    pivot = df_model.pivot_table(
        index='category',
        columns='method',
        values='accuracy',
        aggfunc="mean"
    ).sort_index()

    methods = list(pivot.columns)
    colors = [COLOR_MAP.get(m, 'gray') for m in methods]

    ax = pivot.plot(
        kind='barh',
        figsize=(10, 6),
        title=f"{model_name}: Accuracy pro Kategorie",
        color=colors
    )

    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Kategorie')
    ax.legend(title='Methode')
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"{model_name.replace(' ', '_')}_accuracy_kategorie.png")
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot gespeichert unter: {filename}")
    plt.show()

def compute_accuracy_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet die mittlere Accuracy pro Modell und Methode:
    1. Durchschnitt über Subkategorien pro Kategorie
    2. Durchschnitt über Kategorien
    """

    per_category = (
        df.groupby(['model', 'method', 'category'])
        .agg(acc_per_category=('accuracy', 'mean'))
        .reset_index()
    )

    summary = (
        per_category.groupby(['model', 'method'])
        .agg(mean_accuracy=('acc_per_category', 'mean'))  # Durchschnitt über Kategorien
        .reset_index()
        .pivot(index='model', columns='method', values='mean_accuracy')
        .round(3)
        .fillna(0)
    )

    print("\nMittlere Accuracy pro Modell und Methode:")
    print(summary.to_string())

    return summary

def summarize_accuracy_similarity_per_category_per_model(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Gibt für die angegebene Methode ein DataFrame mit mittlerer Accuracy und Similarity
    je Modell und Kategorie aus, sowie eine Pivot-Tabelle wie bei analyze_by_subcategory_similarity().
    """
    df_method = df[df['method'] == method].copy()

    # Gruppierung: mittlere Accuracy + Similarity je Modell und Hauptkategorie
    grouped = (
        df_method
        .groupby(['model', 'category'])
        .agg(
            mean_accuracy=('accuracy', 'mean'),
            mean_similarity=('similarity', 'mean')
        )
        .reset_index()
        .round(3)
    )

    print(f"\n Mittlere Accuracy und Similarity pro Modell und BATS-Kategorie (Methode: {method}):")
    print(grouped.to_string(index=False))

    return grouped

def plot_heatmap_per_method(df: pd.DataFrame, out_dir: str = 'Plots'):
    """
    Erzeugt für jede Methode eine Heatmap mit Subkategorien (y) vs. Modelle (x).
    Speichert PNGs in out_dir unter dem Namen 'heatmap_<method>.png'.
    """
    os.makedirs(out_dir, exist_ok=True)
    methods = sorted(df['method'].unique())
    for method in methods:
        df_m = df[df['method'] == method]
        pivot = df_m.pivot_table(
            index='subcategory', columns='model', values='accuracy'
        )
        subs = pivot.index.tolist()
        models = pivot.columns.tolist()
        data = pivot.values

        fig, ax = plt.subplots(figsize=(len(models) * 1.2, len(subs) * 0.3))
        c = ax.imshow(data, aspect='auto', cmap='viridis', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(subs)))
        ax.set_yticklabels(subs)

        for i in range(len(subs)):
            for j in range(len(models)):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8, color='white')

        fig.colorbar(c, ax=ax, label='Accuracy')
        ax.set_title(f"{method}: Top-5 Accuracy pro Subkategorie und Modell-Variante")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"heatmap_top5_{method}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap für {method} gespeichert unter: {out_path}")
        plt.close(fig)

def plot_top1_vs_top5_accuracy(df1: pd.DataFrame, df5: pd.DataFrame, model_name: str, out_dir: str = 'plots'):
    """
    Plottet Top-1 vs. Top-5 Accuracy pro BATS-Kategorie und Methode für ein bestimmtes Modell.
    Dabei stehen für eine Methode (z.B. „3CosAdd“) die Top-1- und Top-5-Balken übereinander,
    und die verschiedenen Methoden stehen in einer Kategorie nebeneinander.
    """
    os.makedirs(out_dir, exist_ok=True)

    df1_model = df1[df1['model'] == model_name]
    df5_model = df5[df5['model'] == model_name]

    pivot_top1 = df1_model.groupby(['category', 'method'])['accuracy'].mean().unstack()
    pivot_top5 = df5_model.groupby(['category', 'method'])['accuracy'].mean().unstack()

    categories = sorted(pivot_top1.index)
    methods = list(COLOR_MAP.keys())

    n_cats = len(categories)
    n_methods = len(methods)

    x = np.arange(n_cats)
    bar_width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(methods):
        color = COLOR_MAP[method]

        offset = (i - (n_methods - 1) / 2) * bar_width

        acc_top1 = pivot_top1.get(method, pd.Series(index=categories, data=0.0)).reindex(categories).fillna(0).values
        acc_top5 = pivot_top5.get(method, pd.Series(index=categories, data=0.0)).reindex(categories).fillna(0).values

        x_pos = x + offset

        ax.bar(
            x_pos,
            acc_top1,
            width=bar_width,
            label=f"{method} Top-1",
            color=color,
            alpha=1.0
        )
        ax.bar(
            x_pos,
            acc_top5,
            width=bar_width,
            label=f"{method} Top-5",
            color=color,
            alpha=0.4
        )

    ax.set_xticks(x)
    ax.set_xticklabels([cat.replace('_', ' ').strip() for cat in categories], rotation=45, ha='right')
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Top-1 vs. Top-5 Accuracy pro BATS-Kategorie ({model_name})")
    ax.set_ylim(0, 1.0)

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    ax.legend(unique.values(), unique.keys(), ncol=2, fontsize='small')

    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{model_name.replace(' ', '_')}_top1_vs_top5_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot gespeichert unter: {out_path}")
    plt.show()


def plot_stacked_accuracy_and_similarity(df1: pd.DataFrame):
    """
    Plottet gruppierte Balken: Accuracy (blau-embs/orange-unembs) und Cosine Similarity (grau) pro Modell.
    Zwei Balken pro Modell (embs + unembs), Cosine Similarity im Hintergrund.
    """
    df = df1[df1['method'] == '3CosAvg'].copy()

    model_bases = {
        'gpt2_cleaned': 'GPT-2 Cleaned',
        'pythia_cleaned': 'Pythia Cleaned',
        'pythia_raw': 'Pythia Raw'
    }

    matrix_types = ['embs', 'unembs']
    colors = {
        'embs': '#1f77b4',
        'unembs': '#ff7f0e',
        'similarity': '#d3d3d3'
    }

    categories = [
        '1_Inflectional_morphology',
        '2_Derivational_morphology',
        '3_Encyclopedic_semantics',
        '4_Lexicographic_semantics'
    ]

    acc_values = []
    sim_values = []
    bar_colors = []
    xtick_labels = []
    x_positions = []
    label_positions = []

    group_spacing = 2.5
    bar_spacing = 0.5
    x = 0

    for base_model, model_label in model_bases.items():
        model_group_positions = []
        for matrix in matrix_types:
            model_full = f'{base_model}_{matrix}'
            sub_df = df[df['model'] == model_full]

            acc_per_cat = []
            sim_per_cat = []
            for cat in categories:
                cat_df = sub_df[sub_df['category'] == cat]
                if not cat_df.empty:
                    acc_per_cat.append(cat_df['accuracy'].mean())
                    sim_per_cat.append(cat_df['similarity'].mean())

            if len(acc_per_cat) == 4 and len(sim_per_cat) == 4:
                acc = sum(acc_per_cat) / 4
                sim = sum(sim_per_cat) / 4
                acc_values.append(acc)
                sim_values.append(sim)
                bar_colors.append(colors[matrix])
                x_positions.append(x)
                model_group_positions.append(x)
                x += bar_spacing

        label_positions.append(sum(model_group_positions) / len(model_group_positions))
        xtick_labels.append(model_label)
        x += group_spacing

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x_positions, sim_values, width=0.4, color=colors['similarity'], zorder=1)

    ax.bar(x_positions, acc_values, width=0.4, color=bar_colors, edgecolor='black', linewidth=0.3, zorder=2)

    ax.set_ylabel('Wert', color='black')
    ax.set_ylim(0, 1)
    ax.set_xticks(label_positions)
    ax.set_xticklabels(xtick_labels)
    ax.set_title('Mittlere Top-1 Accuracy und Cosine Similarity pro Modellvariante (3CosAvg)')

    legend_elements = [
        Patch(facecolor=colors['embs'], label='Embeddings'),
        Patch(facecolor=colors['unembs'], label='Unembeddings'),
        Patch(facecolor=colors['similarity'], label='Cosine Similarity')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig("plots/stacked_accuracy_similarity_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    result_folders = [d for d in os.listdir('.') if d.startswith('analogy_results_') and os.path.isdir(d)]
    df1 = pd.read_csv("auswertung_top1_accuracy_similarity.csv")
    df5 = pd.read_csv("auswertung_top5_accuracy_similarity.csv")
    summary_1 = compute_accuracy_per_model(df1)
    summary_2 = summarize_accuracy_similarity_per_category_per_model(df1, method="3CosAvg")
    plot_accuracy_by_category(df1, model_name = "pythia_raw_unembs", out_dir="plots")
    plot_heatmap_per_method(df5, "plots")
    plot_top1_vs_top5_accuracy(df1, df5, model_name="pythia_raw_unembs")
    summarize_accuracy_similarity_per_category_per_model(df1, "3CosAvg")
    plot_stacked_accuracy_and_similarity(df1)



