import os
from shared_token_utils import load_shared_tokens

def filter_bats_to_shared(bats_root, shared_tokens, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    removed_files = []

    for root, _, files in os.walk(bats_root):
        rel = os.path.relpath(root, bats_root)
        out_sub = os.path.join(output_dir, rel)
        os.makedirs(out_sub, exist_ok=True)

        for fn in files:
            if not fn.endswith(".txt"):
                continue
            inp = os.path.join(root, fn)
            out = os.path.join(out_sub, fn)

            lines_written = 0

            with open(inp, "r", encoding="utf-8") as fin, \
                 open(out, "w", encoding="utf-8") as fout:

                for line in fin:
                    if line.startswith(":") or not line.strip():
                        continue

                    a, b_combo = line.strip().split()
                    bs = b_combo.split("/")
                    if a in shared_tokens and all(b in shared_tokens for b in bs):
                        fout.write(f"{a}\t{b_combo}\n")
                        lines_written += 1

            if lines_written == 0:
                os.remove(out)
                removed_files.append(out)

    print(f"Gefilterte BATS-Daten gespeichert in: {output_dir}")
    if removed_files:
        print(f"{len(removed_files)} leere Dateien wurden entfernt:")
        for file in removed_files:
            print(f"   - {file}")

if __name__ == "__main__":
    shared = load_shared_tokens("shared_tokens.txt")
    filter_bats_to_shared("BATS_3.0", shared, "BATS_shared")


