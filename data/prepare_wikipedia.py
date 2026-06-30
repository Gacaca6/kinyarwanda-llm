"""
prepare_wikipedia.py
Ingest the Kinyarwanda Wikipedia article dump into one-sentence-per-line text.

Source (verified 2026-06-30):
  https://dumps.wikimedia.org/rwwiki/<date>/rwwiki-<date>-pages-articles.xml.bz2
  License: CC BY-SA 4.0 + GFDL (text), commercial use OK with attribution.

Same contract as data/prepare_corpus.py: one clean Kinyarwanda sentence per line,
UTF-8, whitespace normalized, Kinyarwanda orthography (n', by' elisions) preserved.
Everything downstream (tokenizer, dataloader) is then identical across sources.

Usage:
  python data/prepare_wikipedia.py                 # download latest known dump + extract
  python data/prepare_wikipedia.py --date 20260601
  python data/prepare_wikipedia.py --max-pages 50  # quick smoke test

Raw dump lands in data/sources/wikipedia/raw/ (gitignored); the cleaned corpus is
written to data/sources/wikipedia/wikipedia_rw.txt.
"""
import argparse
import bz2
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET

import mwparserfromhell

DEFAULT_DATE = "20260601"  # latest verified dump as of 2026-06-30
DUMP_URL = ("https://dumps.wikimedia.org/rwwiki/{date}/"
            "rwwiki-{date}-pages-articles.xml.bz2")
# Wikimedia requires a descriptive User-Agent or it returns HTTP 403.
UA = "kinyarwanda-llm/0.1 (open Kinyarwanda LLM corpus build; contact via GitHub)"

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "sources", "wikipedia")
RAW_DIR = os.path.join(OUT_DIR, "raw")

# --- wikitext cleaning -------------------------------------------------------
_REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*?>.*?</ref>", re.S | re.I)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_LEFTOVER_MARKUP = re.compile(r"\{\{[^{}]*\}\}|\[\[[^\[\]]*\]\]|^[|!{}].*$",
                              re.M)
_WS = re.compile(r"\s+")
_WIKITABLE = re.compile(r"\{\|.*?\|\}", re.S)   # {| ... |} table blocks
_IMG_PX = re.compile(r"\b\d+x\d+\s*px\b|\b\d+\s*px\b", re.I)  # 125px, 321x321px
# Lowercase image-link keywords that strip_code can glue onto caption text
# (e.g. "umugabothumb"). They never occur in Kinyarwanda prose -> safe to drop.
_THUMB = re.compile(r"thumbnail|thumb")
# Sentence boundary: ., !, ? (optionally followed by a quote/bracket) + space.
_SENT = re.compile(r"(?<=[.!?])[\"'»”\])]*\s+")


def clean_wikitext(raw):
    raw = _HTML_COMMENT.sub(" ", raw)
    raw = _REF.sub(" ", raw)
    raw = _WIKITABLE.sub(" ", raw)
    try:
        text = mwparserfromhell.parse(raw).strip_code(normalize=True,
                                                      collapse=True)
    except Exception:
        text = raw
    text = _LEFTOVER_MARKUP.sub(" ", text)
    text = _IMG_PX.sub(" ", text)
    text = _THUMB.sub(" ", text)
    return text


def _is_junk(s):
    """Reject leftover markup residue (table rows, mostly non-letters)."""
    if s[0] in "+|!{}-*#=":               # table/markup line starts
        return True
    if "|" in s:                          # residual table/image markup (thumb|...)
        return True
    letters = sum(c.isalpha() or c.isspace() for c in s)
    return letters / len(s) < 0.65        # too few letters -> not prose


def sentences(text):
    for para in text.split("\n"):
        para = _WS.sub(" ", para).strip()
        if not para:
            continue
        for s in _SENT.split(para):
            s = s.strip()
            if s and not _is_junk(s):
                yield s


def iter_pages(bz2_path, max_pages=0):
    """Stream <page> elements from the bz2 XML; yield article wikitext only."""
    with bz2.open(bz2_path, "rb") as fh:
        n = 0
        # Namespace-agnostic: strip the {http://...} prefix from tags.
        for _, elem in ET.iterparse(fh, events=("end",)):
            tag = elem.tag.split("}")[-1]
            if tag != "page":
                continue
            ns = elem.findtext("./{*}ns")
            redirect = elem.find("./{*}redirect")
            text = elem.findtext("./{*}revision/{*}text")
            elem.clear()  # free memory — dumps are big
            if ns != "0" or redirect is not None or not text:
                continue
            yield text
            n += 1
            if max_pages and n >= max_pages:
                return


def download(url, dest):
    if os.path.exists(dest):
        print(f"  using cached dump: {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as w:
        w.write(r.read())
    print(f"  saved {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=DEFAULT_DATE)
    ap.add_argument("--max-pages", type=int, default=0, help="0 = all")
    ap.add_argument("--min-chars", type=int, default=8,
                    help="drop sentences shorter than this")
    args = ap.parse_args()

    url = DUMP_URL.format(date=args.date)
    dump = os.path.join(RAW_DIR, f"rwwiki-{args.date}-pages-articles.xml.bz2")
    out = os.path.join(OUT_DIR, "wikipedia_rw.txt")

    os.makedirs(OUT_DIR, exist_ok=True)
    download(url, dump)

    seen, n_sent, n_pages = set(), 0, 0
    with open(out, "w", encoding="utf-8") as w:
        for wikitext in iter_pages(dump, args.max_pages):
            n_pages += 1
            for s in sentences(clean_wikitext(wikitext)):
                if len(s) < args.min_chars:
                    continue
                if s in seen:  # exact-line dedup within this source
                    continue
                seen.add(s)
                w.write(s + "\n")
                n_sent += 1
            if n_pages % 500 == 0:
                print(f"  ...{n_pages} articles -> {n_sent} sentences")

    size_mb = os.path.getsize(out) / 1e6
    print(f"Done. {n_pages} articles -> {n_sent} unique sentences "
          f"({size_mb:.1f} MB) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
