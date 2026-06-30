"""
prepare_mbaza.py
Ingest the Mbaza NLP Kinyarwanda monolingual dataset (v01.1) into one-sentence-
per-line text.

Source (verified 2026-06-30):
  https://huggingface.co/datasets/mbazaNLP/kinyarwanda_monolingual_v01.0
  File: data/train-00000-of-00001.parquet, column "text".
  License: CC BY 4.0 (commercial OK WITH attribution). Mbaza NLP Community (2024).
  NB: the maintainer-recommended v01.1 is **gated** (needs HF login) so we use the
  ungated v01.0. Its known issues (some duplicates + a few non-rw docs) are exactly
  what our hash-dedup here + Phase 2 language-ID/near-dedup remove.

Built from Kinyarwanda news (Kigali Today, Igihe), religious + cultural sites,
Wikipedia, and government/legal/educational PDFs -> it OVERLAPS our Wikipedia and
news sources, so cross-source near-dedup in Phase 2 is essential.

Same output contract as the other readers: one clean sentence per line, UTF-8,
whitespace normalized, n'/by' elisions preserved.

Usage:
  python data/prepare_mbaza.py                 # download parquet + extract
  python data/prepare_mbaza.py --max-rows 500  # quick smoke test
"""
import argparse
import hashlib
import os
import re
import sys
import urllib.request

import pyarrow.parquet as pq

REPO = "mbazaNLP/kinyarwanda_monolingual_v01.0"  # v01.1 is gated; v01.0 is open
PARQUET = "data/train-00000-of-00001.parquet"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{PARQUET}"
UA = "kinyarwanda-llm/0.1 (open Kinyarwanda LLM corpus build; contact via GitHub)"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "sources", "mbaza")
RAW_DIR = os.path.join(OUT_DIR, "raw")

_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])[\"'»”\])]*\s+")


def _is_junk(s):
    if s[0] in "+|!{}*#=<>":
        return True
    if "|" in s or "©" in s or "://" in s:
        return True
    letters = sum(c.isalpha() or c.isspace() for c in s)
    return letters / len(s) < 0.6


def sentences(doc):
    doc = doc.replace("\\n", "\n").replace("\\t", " ").replace("\\r", " ")
    for line in doc.split("\n"):
        line = _WS.sub(" ", line).strip()
        if not line:
            continue
        for s in _SENT.split(line):
            s = s.strip()
            if s and not _is_junk(s):
                yield s


def download(url, dest):
    if os.path.exists(dest):
        print(f"  using cached parquet: {dest} ({os.path.getsize(dest)/1e6:.0f} MB)")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url}\n  (276 MB — this takes a few minutes)")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = dest + ".part"
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as w:
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            w.write(chunk)
            done += len(chunk)
            if done % (25 << 20) < (1 << 20):
                print(f"    ...{done/1e6:.0f} MB")
    os.replace(tmp, dest)
    print(f"  saved {dest} ({os.path.getsize(dest)/1e6:.0f} MB)")


def iter_texts(parquet_path, max_rows=0):
    pf = pq.ParquetFile(parquet_path)
    n = 0
    for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
        for text in batch.column("text").to_pylist():
            if not text:
                continue
            yield text
            n += 1
            if max_rows and n >= max_rows:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--min-chars", type=int, default=12)
    args = ap.parse_args()

    parquet = os.path.join(RAW_DIR, os.path.basename(PARQUET))
    out = os.path.join(OUT_DIR, "mbaza_rw.txt")
    os.makedirs(OUT_DIR, exist_ok=True)
    download(URL, parquet)

    seen, n_sent, n_docs, n_bytes = set(), 0, 0, 0
    with open(out, "w", encoding="utf-8") as w:
        for doc in iter_texts(parquet, args.max_rows):
            n_docs += 1
            for s in sentences(doc):
                if len(s) < args.min_chars:
                    continue
                h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
                if h in seen:
                    continue
                seen.add(h)
                line = s + "\n"
                w.write(line)
                n_sent += 1
                n_bytes += len(line.encode("utf-8"))
            if n_docs % 10000 == 0:
                print(f"  ...{n_docs:,} docs -> {n_sent:,} sentences "
                      f"({n_bytes/1e6:.0f} MB)")

    print(f"Done. {n_docs:,} docs -> {n_sent:,} unique sentences "
          f"({n_bytes/1e6:.1f} MB) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
