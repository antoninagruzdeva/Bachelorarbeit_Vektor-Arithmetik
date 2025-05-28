import os
import numpy as np
from gensim.models import KeyedVectors
from transformer_lens import HookedTransformer, HookedEncoder
from shared_token_utils import load_shared_tokens, normalize_token


def load_model(model_name: str, cleaned: bool):
    if model_name.startswith("bert"):
        return HookedEncoder.from_pretrained(model_name)
    return (
        HookedTransformer.from_pretrained(model_name)
        if cleaned else
        HookedTransformer.from_pretrained_no_processing(model_name)
    )


def extract_matrix(model, matrix_type: str) -> np.ndarray:
    """Gibt die Embedding- oder Unembedding-Matrix des Modells zurück."""
    if matrix_type == "embedding":
        return model.W_E.detach().cpu().numpy().astype(np.float32)
    elif matrix_type == "unembedding":
        return model.W_U.detach().cpu().numpy().T.astype(np.float32)
    else:
        raise ValueError("matrix_type muss 'embedding' oder 'unembedding' sein.")


def extract_and_save(model_name: str, cleaned: bool, matrix_type: str, shared_tokens_path="shared_tokens.txt"):
    shared_tokens = load_shared_tokens(shared_tokens_path)
    model = load_model(model_name, cleaned)
    vocab = model.tokenizer.get_vocab()
    matrix = extract_matrix(model, matrix_type)

    norm_to_id = {}
    for token, idx in vocab.items():
        norm = normalize_token(model_name, token)
        if norm in shared_tokens and norm is not None and norm not in norm_to_id:
            norm_to_id[norm] = idx

    missing = [tok for tok in shared_tokens if tok not in norm_to_id]
    if missing:
        raise ValueError(f"{len(missing)} shared_tokens fehlen im Modellvokabular: {model_name} ({matrix_type})")

    filtered_ids = [norm_to_id[tok] for tok in shared_tokens]
    filtered_matrix = matrix[filtered_ids]

    kv = KeyedVectors(vector_size=filtered_matrix.shape[1])
    kv.add_vectors(shared_tokens, filtered_matrix)

    cleaned_flag = "cleaned" if cleaned else "raw"
    model_folder = f"{model_name.replace('-', '_')}_{cleaned_flag}_{matrix_type}s" if not model_name.startswith("bert") else f"{model_name.replace('-', '_')}_{matrix_type}s"

    os.makedirs(model_folder, exist_ok=True)
    out_path = os.path.join(model_folder, f"{model_folder}.bin")
    kv.save_word2vec_format(out_path, binary=True)
    print(f"Gespeichert: {out_path}")

if __name__ == "__main__":
    extract_and_save("gpt2-small", cleaned=True,  matrix_type="embedding")
    extract_and_save("gpt2-small", cleaned=True,  matrix_type="unembedding")
    extract_and_save("gpt2-small", cleaned=False, matrix_type="embedding")
    extract_and_save("pythia-410m", cleaned=True,  matrix_type="embedding")
    extract_and_save("pythia-410m", cleaned=True,  matrix_type="unembedding")
    extract_and_save("pythia-410m", cleaned=False, matrix_type="embedding")
    extract_and_save("pythia-410m", cleaned=False, matrix_type="unembedding")
    extract_and_save("bert-base-uncased", cleaned=True, matrix_type="embedding")

