"""
compare_tokenizers.py
Measures how efficiently each tokenizer encodes Kinyarwanda text.

Metric: "fertility" = tokens produced per whitespace word.
Lower is better: fewer tokens per word means each token carries more meaning,
the model sees longer effective context, and training/inference are cheaper.
"""
import sys
import os
from tokenizers import Tokenizer

# Use the real GPT-2 BPE encoder bundled with the book repo (fully offline).
_BOOK_BPE = "/home/claude/LLMs-from-scratch/ch02/02_bonus_bytepair-encoder"
sys.path.insert(0, _BOOK_BPE)
from bpe_openai_gpt2 import get_encoder  # noqa: E402


def stats(name, encode_fn, lines):
    n_tokens = n_words = n_chars = 0
    for ln in lines:
        n_tokens += len(encode_fn(ln))
        n_words += len(ln.split())
        n_chars += len(ln)
    print(f"{name:<28} tokens={n_tokens:>7}  "
          f"tokens/word={n_tokens / n_words:5.2f}  "
          f"chars/token={n_chars / n_tokens:5.2f}")
    return n_tokens / n_words


def main():
    lines = [l.strip() for l in open(
        "../data/kinyarwanda_corpus.txt", encoding="utf-8") if l.strip()]

    gpt2 = get_encoder(model_name="gpt2_model", models_dir=_BOOK_BPE)
    kin = Tokenizer.from_file("kinyarwanda_bpe/tokenizer.json")

    print(f"Evaluated on {len(lines)} Kinyarwanda sentences\n")
    f_gpt2 = stats("GPT-2 (English BPE, book default)",
                   lambda s: gpt2.encode(s), lines)
    f_kin = stats("Kinyarwanda BPE (ours)",
                  lambda s: kin.encode(s).ids, lines)

    print(f"\n=> Kinyarwanda tokenizer is "
          f"{f_gpt2 / f_kin:.2f}x more efficient on Kinyarwanda "
          f"({(1 - f_kin / f_gpt2) * 100:.0f}% fewer tokens per word).")

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
