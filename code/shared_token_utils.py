def load_shared_tokens(path="shared_tokens.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def normalize_token(model_name: str, token: str):
    if "gpt2" in model_name or "pythia" in model_name:
        return token.lstrip("Ġ").lower()
    elif "bert" in model_name:
        if token.startswith("##"):
            return None
        return token
    else:
        return token.lower()
