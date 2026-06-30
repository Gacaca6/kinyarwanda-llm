"""
sweep_tokenizer.py — Phase 3: scale the Kinyarwanda BPE tokenizer to the real corpus.

Trains byte-level BPE on data/processed/train.txt at several vocab sizes, then
measures *fertility* (tokens per whitespace word) on the held-out data/processed/val.txt.
Lower fertility = each token carries more meaning = longer effective context and
cheaper training/inference. We also report GPT-2 (tiktoken) fertility as the baseline
the project is improving on.

Bigger vocab lowers fertility but inflates the model's embedding + output head
(vocab x emb_dim, counted twice), so the goal is the best *trade-off*, not the lowest
fertility outright.

Run from the tokenizer/ dir:
  python sweep_tokenizer.py
Outputs trained tokenizers to tokenizer/sweep/vocab_<n>/ and a report to
../eval/tokenizer_sweep.md.
"""
import os
import sys
import time

from tokenizers import ByteLevelBPETokenizer
import tiktoken

sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN = os.path.join(ROOT, "data", "processed", "train.txt")
VAL = os.path.join(ROOT, "data", "processed", "val.txt")
SWEEP_DIR = os.path.join(HERE, "sweep")
REPORT = os.path.join(ROOT, "eval", "tokenizer_sweep.md")

VOCAB_SIZES = [16000, 32000, 50000]
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
SAMPLE_WORDS = ["ntibazabikora", "turabashimira", "abanyarwanda",
                "uburezi", "umuganda", "icyorezo", "umunyarwanda",
                "ubwiyunge", "guhanga"]


def load_val_words(path, limit_lines=200000):
    """Read held-out text; return (list_of_lines, total_word_count)."""
    lines, words = [], 0
    with open(path, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i >= limit_lines:
                break
            ln = ln.strip()
            if ln:
                lines.append(ln)
                words += len(ln.split())
    return lines, words


def fertility(encode_fn, lines, n_words):
    n_tokens = sum(len(encode_fn(ln)) for ln in lines)
    n_chars = sum(len(ln) for ln in lines)
    return n_tokens / n_words, n_chars / n_tokens, n_tokens


def main():
    if not os.path.exists(TRAIN):
        sys.exit(f"Missing {TRAIN} — run data/clean_corpus.py first (Phase 2).")
    os.makedirs(SWEEP_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)

    val_lines, val_words = load_val_words(VAL)
    print(f"Held-out val: {len(val_lines):,} lines, {val_words:,} words\n")

    # Baseline: GPT-2 (English BPE) on the same Kinyarwanda val text.
    gpt2 = tiktoken.get_encoding("gpt2")
    f_gpt2, c_gpt2, _ = fertility(lambda s: gpt2.encode(s), val_lines, val_words)
    print(f"GPT-2 baseline      fertility={f_gpt2:5.2f}  chars/token={c_gpt2:5.2f}\n")

    rows = []          # (vocab, tokenizer, fertility, chars/token, secs)
    for v in VOCAB_SIZES:
        out = os.path.join(SWEEP_DIR, f"vocab_{v}")
        os.makedirs(out, exist_ok=True)
        t0 = time.time()
        tok = ByteLevelBPETokenizer()
        tok.train(files=[TRAIN], vocab_size=v, min_frequency=2,
                  special_tokens=SPECIAL_TOKENS)
        tok.save(os.path.join(out, "tokenizer.json"))
        tok.save_model(out)
        secs = time.time() - t0
        f_k, c_k, _ = fertility(lambda s: tok.encode(s).ids, val_lines, val_words)
        rows.append((v, tok, f_k, c_k, secs))
        print(f"vocab={v:<6} fertility={f_k:5.2f}  chars/token={c_k:5.2f}  "
              f"vs GPT-2 {f_gpt2/f_k:.2f}x  (trained {secs:.0f}s)")

    # Write report
    with open(REPORT, "w", encoding="utf-8") as r:
        r.write("# Tokenizer sweep (Phase 3)\n\n")
        r.write(f"Trained on `data/processed/train.txt`, evaluated on "
                f"`data/processed/val.txt` ({val_words:,} words).\n\n")
        r.write("| Tokenizer | vocab | tokens/word (fertility) | chars/token | "
                "vs GPT-2 |\n|---|---|---|---|---|\n")
        r.write(f"| GPT-2 (English BPE) | 50257 | {f_gpt2:.2f} | {c_gpt2:.2f} | 1.00x |\n")
        for v, _tok, f_k, c_k, _s in rows:
            r.write(f"| Kinyarwanda BPE | {v} | {f_k:.2f} | {c_k:.2f} | "
                    f"{f_gpt2/f_k:.2f}x |\n")
        r.write("\nLower fertility is better. Bigger vocab lowers fertility but "
                "enlarges the model's embedding + output head (vocab x emb_dim x2).\n\n")
        r.write("## Example: agglutinative words, token counts by vocab\n\n```\n")
        for w in SAMPLE_WORDS:
            parts = [f"{w:<16} GPT-2={len(gpt2.encode(w))}"]
            for v, tok, *_ in rows:
                parts.append(f"v{v}={len(tok.encode(w).ids)}")
            r.write("  ".join(parts) + "\n")
        r.write("```\n")
    print(f"\nReport -> {REPORT}")


if __name__ == "__main__":
    sys.exit(main())
