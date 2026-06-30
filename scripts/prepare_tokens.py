"""
prepare_tokens.py — tokenize the Phase-2 corpus into flat uint16 token arrays.

Why a binary token file: the book's in-memory GPTDataset materializes one tensor per
sliding-window chunk, which blows up RAM on a 178M-token corpus. The standard
small-LLM practice (nanoGPT) is to pre-tokenize once into a flat array and memmap it,
then sample random windows at train time. 32k vocab fits in uint16 (0..65535).

Each line (a sentence, post-Phase-2) is encoded and followed by the <|endoftext|>
token, so the model learns sentence boundaries. (Phase 2 shuffled at sentence level,
so adjacent lines are unrelated — EOT separation is the honest framing. Document-level
packing is a future improvement.)

Run from project root:
  python -m scripts.prepare_tokens                 # all splits, full
  python -m scripts.prepare_tokens --limit-lines 100000   # quick/smoke train.bin
Outputs data/processed/{train,val,test}.bin (gitignored).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.data_loader import KinyaTokenizer  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROC = "data/processed"
TOK_PATH = "tokenizer/kinyarwanda_bpe/tokenizer.json"
FLUSH_EVERY = 2_000_000  # token-buffer flush size


def encode_file(in_path, out_path, tok, eot, limit_lines=0):
    if not os.path.exists(in_path):
        print(f"  skip (missing): {in_path}")
        return 0
    buf, total = [], 0
    with open(out_path, "wb") as out:
        for i, line in enumerate(open(in_path, encoding="utf-8")):
            if limit_lines and i >= limit_lines:
                break
            line = line.strip()
            if not line:
                continue
            ids = tok.encode(line)
            ids.append(eot)
            buf.extend(ids)
            if len(buf) >= FLUSH_EVERY:
                np.asarray(buf, dtype=np.uint16).tofile(out)
                total += len(buf)
                buf.clear()
            if (i + 1) % 500000 == 0:
                print(f"    {os.path.basename(in_path)}: {i+1:,} lines, "
                      f"{total + len(buf):,} tokens")
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(out)
            total += len(buf)
    print(f"  {os.path.basename(out_path)}: {total:,} tokens "
          f"({os.path.getsize(out_path)/1e6:.0f} MB)")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-lines", type=int, default=0,
                    help="cap train.txt lines (val/test always full); 0 = all")
    args = ap.parse_args()

    tok = KinyaTokenizer(TOK_PATH)
    eot = tok.eot_id
    assert tok.vocab_size <= 65536, "vocab too big for uint16"
    print(f"Tokenizer vocab={tok.vocab_size}, eot_id={eot}\n")

    encode_file(f"{PROC}/train.txt", f"{PROC}/train.bin", tok, eot, args.limit_lines)
    encode_file(f"{PROC}/val.txt", f"{PROC}/val.bin", tok, eot)
    encode_file(f"{PROC}/test.txt", f"{PROC}/test.bin", tok, eot)


if __name__ == "__main__":
    sys.exit(main())
