"""
compare_tokenizers.py
Measures how efficiently each tokenizer encodes Kinyarwanda text.

Metric: "fertility" = tokens produced per whitespace word.
Lower is better: fewer tokens per word means each token carries more meaning,
the model sees longer effective context, and training/inference are cheaper.

Baseline is GPT-2's English BPE via `tiktoken` (cross-platform; replaces the book's
bundled offline encoder, which lived at a Linux-only path). Evaluated on held-out
Kinyarwanda (data/processed/val.txt) if present, else the demo corpus.
"""
import os
import sys

from tokenizers import Tokenizer
import tiktoken

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 consoles

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VAL = os.path.join(ROOT, "data", "processed", "val.txt")
DEMO = os.path.join(ROOT, "data", "kinyarwanda_corpus.txt")
KIN_TOK = os.path.join(HERE, "kinyarwanda_bpe", "tokenizer.json")


def stats(name, encode_fn, lines):
    n_tokens = n_words = n_chars = 0
    for ln in lines:
        n_tokens += len(encode_fn(ln))
        n_words += len(ln.split())
        n_chars += len(ln)
    print(f"{name:<30} tokens={n_tokens:>9,}  "
          f"tokens/word={n_tokens / n_words:5.2f}  "
          f"chars/token={n_chars / n_tokens:5.2f}")
    return n_tokens / n_words


def main():
    path = VAL if os.path.exists(VAL) else DEMO
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]

    gpt2 = tiktoken.get_encoding("gpt2")
    kin = Tokenizer.from_file(KIN_TOK)

    print(f"Evaluated on {len(lines):,} Kinyarwanda sentences "
          f"({os.path.relpath(path, ROOT)})\n")
    f_gpt2 = stats("GPT-2 (English BPE, tiktoken)", lambda s: gpt2.encode(s), lines)
    f_kin = stats(f"Kinyarwanda BPE (vocab {kin.get_vocab_size()})",
                  lambda s: kin.encode(s).ids, lines)

    print(f"\n=> Kinyarwanda tokenizer is {f_gpt2 / f_kin:.2f}x more efficient on "
          f"Kinyarwanda ({(1 - f_kin / f_gpt2) * 100:.0f}% fewer tokens per word).")

    print("\nExample - agglutinative words split by each tokenizer:")
    samples = ["ntibazabikora", "turabashimira", "abanyarwanda",
               "uburezi", "umuganda", "icyorezo"]
    for w in samples:
        g = [gpt2.decode([t]) for t in gpt2.encode(w)]
        k = [kin.id_to_token(t) for t in kin.encode(w).ids]
        print(f"  {w:<16} GPT-2 ({len(g)}): {g}")
        print(f"  {'':<16} ours  ({len(k)}): {k}")


if __name__ == "__main__":
    main()
