"""
prepare_madlad.py
Ingest the Kinyarwanda (rw) split of MADLAD-400 into one-sentence-per-line text.

Source (verified 2026-06-30):
  https://huggingface.co/datasets/allenai/MADLAD-400
  File: data/rw/rw_clean_0000.jsonl.gz  (277.7 MB compressed, ~1 GB text)
  License: CC-BY-4.0 (commercial OK WITH attribution); underlying text is Common
  Crawl, so original page copyrights still apply. Cite Kudugunta et al.,
  MADLAD-400 (2023). MADLAD-400 is document-level Common Crawl text, language-ID'd;
  the "clean" split has extra quality filtering (still web-noisy).

We bypass `datasets` on purpose: datasets>=4 removed loading-script support and
MADLAD ships a script. So we pull the raw .jsonl.gz shard directly from the Hub
(one JSON object per line, field "text") and stream-process it.

Same output contract as the other readers: one clean Kinyarwanda sentence per line,
UTF-8, whitespace normalized, n'/by' elisions preserved. Heavy cleaning (language
re-ID, boilerplate/site-name stripping, near-dedup) is Phase 2, not here.

Usage:
  python data/prepare_madlad.py                 # download shard + extract (cap 500 MB out)
  python data/prepare_madlad.py --max-docs 200  # quick smoke test
  python data/prepare_madlad.py --max-out-mb 0  # no output cap (process everything)
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import urllib.request

REPO = "allenai/MADLAD-400"
# rw shards: data/rw/rw_clean_0000.jsonl.gz (278 MB), rw_noisy_0000.jsonl.gz (737 MB)
SHARD_TMPL = "data/rw/rw_{split}_0000.jsonl.gz"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
UA = "kinyarwanda-llm/0.1 (open Kinyarwanda LLM corpus build; contact via GitHub)"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "sources", "madlad")
RAW_DIR = os.path.join(OUT_DIR, "raw")

_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])[\"'»”\])]*\s+")


def _is_junk(s):
    """Reject obvious web boilerplate / markup residue."""
    if s[0] in "+|!{}*#=<>":
        return True
    if "|" in s or "©" in s or "://" in s:   # nav separators, copyright, URLs
        return True
    letters = sum(c.isalpha() or c.isspace() for c in s)
    return letters / len(s) < 0.6


def sentences(doc):
    # MADLAD encodes in-document line breaks as the literal 2-char sequence "\n"
    # (escaped in the source JSON), so they survive json.loads as backslash+n.
    doc = doc.replace("\\n", "\n").replace("\\t", " ").replace("\\r", " ")
    for line in doc.split("\n"):
        line = _WS.sub(" ", line).strip()
        if not line:
            continue
        for s in _SENT.split(line):
            s = s.strip()
            if s and not _is_junk(s):
                yield s


def iter_docs(gz_path, max_docs=0):
    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        n = 0
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                text = json.loads(raw).get("text", "")
            except json.JSONDecodeError:
                continue
            if text:
                yield text
                n += 1
                if max_docs and n >= max_docs:
                    return


def download(url, dest):
    if os.path.exists(dest):
        print(f"  using cached shard: {dest} ({os.path.getsize(dest)/1e6:.0f} MB)")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url}\n  (large file — this takes a few minutes)")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = dest + ".part"
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as w:
        done = 0
        while True:
            chunk = r.read(1 << 20)  # 1 MB
            if not chunk:
                break
            w.write(chunk)
            done += len(chunk)
            if done % (25 << 20) < (1 << 20):
                print(f"    ...{done/1e6:.0f} MB")
    os.replace(tmp, dest)
    print(f"  saved {dest} ({os.path.getsize(dest)/1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["clean", "noisy"], default="clean")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all")
    ap.add_argument("--max-out-mb", type=int, default=0,
                    help="stop after this many MB written (0 = unlimited)")
    ap.add_argument("--min-chars", type=int, default=12)
    args = ap.parse_args()

    shard_rel = SHARD_TMPL.format(split=args.split)
    url = RESOLVE + shard_rel
    shard = os.path.join(RAW_DIR, os.path.basename(shard_rel))
    out = os.path.join(OUT_DIR, f"madlad_rw_{args.split}.txt")
    os.makedirs(OUT_DIR, exist_ok=True)
    download(url, shard)

    cap = args.max_out_mb * 1_000_000
    seen, n_sent, n_docs, n_bytes = set(), 0, 0, 0
    with open(out, "w", encoding="utf-8") as w:
        for doc in iter_docs(shard, args.max_docs):
            n_docs += 1
            for s in sentences(doc):
                if len(s) < args.min_chars:
                    continue
                h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
                if h in seen:            # exact-line dedup (hash -> bounded memory)
                    continue
                seen.add(h)
                line = s + "\n"
                w.write(line)
                n_sent += 1
                n_bytes += len(line.encode("utf-8"))
            if n_docs % 20000 == 0:
                print(f"  ...{n_docs:,} docs -> {n_sent:,} sentences "
                      f"({n_bytes/1e6:.0f} MB)")
            if cap and n_bytes >= cap:
                print(f"  reached output cap ({args.max_out_mb} MB) — stopping.")
                break

    print(f"Done. {n_docs:,} docs -> {n_sent:,} unique sentences "
          f"({n_bytes/1e6:.1f} MB) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
