"""
prepare_corpus.py
Reconstructs running Kinyarwanda text from CoNLL-format NER files
(token <TAB> tag, blank line = sentence boundary) into a plain-text corpus.

This is the *data ingestion* stage. The same structure is how you fold in the
much larger corpora listed in the README (Wikipedia rw, news dumps, the Bible,
the OSCAR / CC-100 rw split, etc.): each new source only needs a reader that
yields one clean sentence per line, then everything downstream is identical.
"""
import sys
import re
import glob


# Light detokenizer: re-attach punctuation that NER tokenization split off.
_NO_SPACE_BEFORE = set(list(".,;:!?)]}%") + ["”", "’", "…"])
_NO_SPACE_AFTER = set(list("([{") + ["“", "‘"])


def detok(tokens):
    out = ""
    for i, t in enumerate(tokens):
        if i == 0:
            out = t
        elif t in _NO_SPACE_BEFORE or (out and out[-1] in _NO_SPACE_AFTER):
            out += t
        else:
            out += " " + t
    return out


def read_conll(path):
    sent = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if sent:
                    yield detok(sent)
                    sent = []
                continue
            tok = line.split()[0]  # first column = surface token
            sent.append(tok)
    if sent:
        yield detok(sent)


def main(inputs, out_path):
    seen, n = set(), 0
    with open(out_path, "w", encoding="utf-8") as w:
        for path in inputs:
            for sent in read_conll(path):
                s = re.sub(r"\s+", " ", sent).strip()
                if len(s) < 8:        # drop fragments
                    continue
                if s in seen:         # dedup exact lines
                    continue
                seen.add(s)
                w.write(s + "\n")
                n += 1
    print(f"Wrote {n} unique sentences -> {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        inputs, out_path = args[:-1], args[-1]
    else:
        inputs = glob.glob(
            "../../masakhane-ner/transfer_corpus/news/kin/*.txt"
        )
        out_path = "kinyarwanda_corpus.txt"
    main(inputs, out_path)
