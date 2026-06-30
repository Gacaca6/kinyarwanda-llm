"""
clean_corpus.py — Phase 2: turn the raw per-source corpora into ONE clean, deduped,
shuffled Kinyarwanda corpus with train/val/test splits.

Pipeline (CLAUDE.md Phase 2):
  1. Gather every cleaned per-source file: data/sources/<name>/*.txt
  2. Normalize: Unicode NFC, collapse whitespace, strip control chars.
  3. Language filter: keep Kinyarwanda, drop foreign lines.
     - fastText lid.176 was tested and CANNOT detect Kinyarwanda (it misfires to
       hr/id/sw/en at <0.3 confidence on real rw, while nailing en/fr at >0.85).
       So we use the manual's sanctioned fallback: the KINNEWS Kinyarwanda stopword
       list (data/lang_id/listed.txt) as a *negative* filter — drop a line only when
       it is confidently foreign (>=2 English/French function words AND zero
       Kinyarwanda function words), plus a non-Latin-script drop. Our sources are
       already ~95%+ rw, so this precisely removes the foreign minority (~1%) without
       deleting short genuine Kinyarwanda sentences.
  4. Dedup: collapse on a normalized key (casefold + letters/digits only + single
     spaces). This removes exact dups, trivial punctuation/case variants, AND the
     heavy cross-source overlap (mbaza re-includes Wikipedia + news that MADLAD also
     crawled). NB: this is normalized-exact dedup; semantic/MinHash near-dedup is a
     future refinement (datasketch) — documented, not yet applied.
  5. Shuffle (offset permutation — memory-light) and split 98/1/1 into
     data/processed/{train,val,test}.txt.

Run: python data/clean_corpus.py
Outputs go to data/processed/ (gitignored — large).
"""
import glob
import hashlib
import os
import random
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_GLOB = os.path.join(HERE, "sources", "*", "*.txt")
OUT_DIR = os.path.join(HERE, "processed")
STOP_PATH = os.path.join(HERE, "lang_id", "listed.txt")

SEED = 123
MIN_CHARS = 12
SPLIT = (0.98, 0.01, 0.01)  # train / val / test

# Foreign (English + French) high-frequency function words for the negative filter.
EN = {"the", "and", "of", "to", "in", "is", "are", "was", "were", "for", "with",
      "that", "this", "you", "your", "on", "at", "by", "from", "as", "it", "be",
      "or", "an", "have", "has", "will", "not", "but", "they", "we", "she", "his",
      "her", "all", "can", "download", "free", "online", "click", "copyright",
      "rights", "reserved", "privacy", "policy", "terms", "page", "home", "more"}
FR = {"le", "la", "les", "des", "du", "une", "et", "est", "pour", "dans", "que",
      "qui", "sur", "avec", "aux", "cette", "par", "plus", "pas", "son", "ses",
      "nous", "vous", "leur", "ete", "etre", "ou", "comme", "mais", "tout", "aussi",
      "sont", "cest", "nest"}
FOREIGN = EN | FR

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_TOK = re.compile(r"[^\W\d_]+", re.UNICODE)      # letter runs (apostrophes split)
_NONKEY = re.compile(r"[^0-9a-zÀ-ɏ]+")  # for the dedup key


def load_stopwords():
    return set(w.strip().lower() for w in open(STOP_PATH, encoding="utf-8")
              if w.strip())


def normalize(line):
    line = unicodedata.normalize("NFC", line)
    line = _CTRL.sub(" ", line)
    return _WS.sub(" ", line).strip()


def keep(line, stop):
    if len(line) < MIN_CHARS:
        return False
    alpha = sum(c.isalpha() for c in line)
    if alpha == 0:
        return False
    latin = sum(1 for c in line if c.isascii() and c.isalpha())
    if latin / alpha < 0.5:               # non-Latin script -> not Kinyarwanda
        return False
    if alpha / len(line) < 0.5:           # mostly digits/symbols -> boilerplate
        return False
    toks = [t.lower() for t in _TOK.findall(line)]
    k = sum(1 for t in toks if t in stop)
    f = sum(1 for t in toks if t in FOREIGN)
    if f >= 2 and k == 0:                  # confidently foreign
        return False
    return True


def dedup_key(line):
    # 128-bit: at tens of millions of lines, 64-bit would expect ~1 random
    # collision (birthday bound) -> a spurious dedup drop. 128-bit makes that nil.
    return hashlib.blake2b(
        _NONKEY.sub(" ", line.casefold()).strip().encode("utf-8"),
        digest_size=16).digest()


def main():
    try:                                   # Windows consoles default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    files = sorted(glob.glob(SRC_GLOB))
    if not files:
        sys.exit(f"No source files matched {SRC_GLOB}")
    os.makedirs(OUT_DIR, exist_ok=True)
    stop = load_stopwords()
    tmp = os.path.join(OUT_DIR, "_clean.tmp")

    print(f"Sources ({len(files)}):")
    for f in files:
        print(f"  {f}")

    raw_lines = raw_words = 0
    kept = kept_words = 0
    seen = set()
    offsets = []

    # Pass 1: normalize -> language filter -> dedup -> write tmp, record byte offsets.
    # Binary I/O so recorded offsets are true byte positions (robust on Windows).
    with open(tmp, "wb") as out:
        for path in files:
            for line in open(path, encoding="utf-8"):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                raw_lines += 1
                raw_words += len(line.split())
                norm = normalize(line)
                if not keep(norm, stop):
                    continue
                key = dedup_key(norm)
                if key in seen:
                    continue
                seen.add(key)
                offsets.append(out.tell())
                out.write(norm.encode("utf-8") + b"\n")
                kept += 1
                kept_words += len(norm.split())

    n = len(offsets)
    print(f"\nraw lines        : {raw_lines:>12,}  words {raw_words:>14,}")
    print(f"after filter+dedup: {n:>12,}  words {kept_words:>14,}  "
          f"({n/raw_lines*100:.1f}% of lines kept)")
    print(f"removed (foreign/junk/duplicate): {raw_lines - n:,} lines")

    # Shuffle via offset permutation (memory-light), then split.
    random.seed(SEED)
    random.shuffle(offsets)
    n_test = int(n * SPLIT[2])
    n_val = int(n * SPLIT[1])
    splits = {
        "test.txt": offsets[:n_test],
        "val.txt": offsets[n_test:n_test + n_val],
        "train.txt": offsets[n_test + n_val:],
    }

    with open(tmp, "rb") as src:
        for fname, offs in splits.items():
            words = 0
            with open(os.path.join(OUT_DIR, fname), "w",
                      encoding="utf-8", newline="\n") as w:
                for off in offs:
                    src.seek(off)
                    ln = src.readline().decode("utf-8")
                    w.write(ln)
                    words += len(ln.split())
            mb = os.path.getsize(os.path.join(OUT_DIR, fname)) / 1e6
            print(f"  {fname:10} lines {len(offs):>11,}  words {words:>13,}  "
                  f"~tokens {int(words*1.56):>13,}  ({mb:.0f} MB)")

    os.remove(tmp)
    print(f"\nDone -> {OUT_DIR}/  (train/val/test). "
          f"Estimated total tokens ~{int(kept_words*1.56):,} (x1.56 words).")


if __name__ == "__main__":
    sys.exit(main())
