
import os
from gensim.models import KeyedVectors
from shared_token_utils import load_shared_tokens, normalize_token

def extract_and_save_word2vec(word2vec_path: str, shared_tokens_path: str = "shared_tokens.txt"):
    shared_tokens = load_shared_tokens(shared_tokens_path)
    kv_full = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    norm_to_id = {}
    for idx, token in enumerate(kv_full.index_to_key):
        norm = normalize_token("word2vec", token)
        if norm in shared_tokens and norm not in norm_to_id:
            norm_to_id[norm] = idx

    filtered_ids = [norm_to_id[tok] for tok in shared_tokens]
    filtered_matrix = kv_full.vectors[filtered_ids]

    kv = KeyedVectors(vector_size=filtered_matrix.shape[1])
    kv.add_vectors(shared_tokens, filtered_matrix)

    out_dir = "word2vec_embs"
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "word2vec_embs.bin")
    kv.save_word2vec_format(bin_path, binary=True)

    print(f"Gefilterte Word2Vec-Matrix gespeichert unter: {bin_path}")

if __name__ == "__main__":
    # Passe diesen Pfad bei Bedarf an
    extract_and_save_word2vec("GoogleNews-vectors-negative300.bin")
