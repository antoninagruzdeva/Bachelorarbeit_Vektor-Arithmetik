# code/extract_shared_tokens.py

from transformer_lens import HookedTransformer, HookedEncoder
from gensim.models import KeyedVectors
import os

def load_vocabularies():
    models = {
        "gpt2": HookedTransformer.from_pretrained_no_processing("gpt2-small"),
        "pythia": HookedTransformer.from_pretrained_no_processing("pythia-410m"),
        "bert": HookedEncoder.from_pretrained("bert-base-uncased"),
        "word2vec": KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    }

    vocabs = {
        "gpt2": models["gpt2"].tokenizer.get_vocab(),
        "pythia": models["pythia"].tokenizer.get_vocab(),
        "bert": models["bert"].tokenizer.get_vocab(),
        "word2vec": models["word2vec"].key_to_index,
    }

    return vocabs

def clean_vocab(model, token):
    if model in ("gpt2", "pythia"):
        norm = token.lstrip("Ġ").lower()
    elif model == "bert":
        if token.startswith("##"):
            return None
        norm = token
    else:
        norm = token.lower()

    if norm.isalpha() and len(norm) >= 3:
        return norm
    return None

def compute_shared_tokens(vocabs):
    cleaned_vocabs = {}
    for model, vocab in vocabs.items():
        cleaned = {clean_vocab(model, token) for token in vocab if clean_vocab(model, token)}
        cleaned_vocabs[model] = cleaned
        print(f"{model}: {len(cleaned)} Tokens nach Cleaning")

    shared = set.intersection(*cleaned_vocabs.values())
    print(f"\nAnzahl gemeinsamer Tokens: {len(shared)}")
    return shared

def save_shared_tokens(shared_tokens, path="shared_tokens.txt"):
    with open(path, "w", encoding="utf-8") as f:
        for tok in sorted(shared_tokens):
            f.write(tok + "\n")
    print(f"Shared tokens gespeichert in: {path}")

if __name__ == "__main__":
    vocabs = load_vocabularies()
    shared_tokens = compute_shared_tokens(vocabs)
    save_shared_tokens(shared_tokens)
