# Data Sources — provenance log

Every corpus added to this project MUST be recorded here before use
(CLAUDE.md §6 "Track data provenance"). Verify each link and its license by
fetching the real page **before** downloading (CLAUDE.md §10). Do not invent
URLs or license terms.

## Record template (copy per source)

### <source name>
- **URL:** <link verified on YYYY-MM-DD>
- **License:** <exact license + commercial/non-commercial>
- **Date downloaded:** YYYY-MM-DD
- **Raw size:** <MB/GB, # docs/sentences>
- **Local path:** `data/sources/<name>/`
- **Reader script:** `data/prepare_<name>.py`
- **Cleaning applied:** <normalization, lang-ID filter, dedup, etc.>
- **Notes:** <caveats, encoding issues, anything for the next session>

---

## Tooling resources (not training text)

### KINNEWS Kinyarwanda stopword list (used for language-ID in Phase 2)
- **URL:** https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus
  file: `stopwords/Kinyarwanda/listed.txt` (verified 2026-06-30)
- **Use:** 80 high-frequency Kinyarwanda function words → drives the negative
  language filter in `data/clean_corpus.py`. Stored at `data/lang_id/listed.txt`
  (committed; tiny). Cite Niyongabo Rubungo et al., *KINNEWS and KIRNEWS* (COLING 2020).
- **Why not fastText lid.176:** tested — it cannot detect Kinyarwanda (misfires to
  hr/id/sw/en at <0.3 confidence on real rw text, while detecting en/fr at >0.85).
  So we use the KINNEWS stopwords as the manual's sanctioned heuristic instead.

## Sources logged

### Mbaza NLP — Kinyarwanda monolingual (v01.0)
- **URL:** https://huggingface.co/datasets/mbazaNLP/kinyarwanda_monolingual_v01.0
  shard: .../resolve/main/data/train-00000-of-00001.parquet (verified 2026-06-30)
- **License:** **CC BY 4.0** — commercial OK **with attribution**. Mbaza NLP
  Community (2024).
- **Date downloaded:** 2026-06-30
- **Raw size:** ~191 MB parquet, column `text`; 78,733 documents.
- **Local path:** `data/sources/mbaza/raw/train-00000-of-00001.parquet` (gitignored)
  → `data/sources/mbaza/mbaza_rw.txt` (cleaned)
- **Reader script:** `data/prepare_mbaza.py` (pyarrow batched read of `text`)
- **Cleaning applied:** same pipeline as MADLAD (unescape → sentence-split → boilerplate
  filter → hash-dedup → one sentence/line, UTF-8, elisions preserved).
- **Output:** **78,733 docs → 966,610 unique sentences, ~21.6M words, ≈34M tokens, 159 MB.**
- **Notes:** Used **v01.0** because the maintainer-recommended **v01.1 is gated**
  (HF login required). v01.0 has known duplicates + a few non-rw docs — handled by
  hash-dedup here and Phase 2 language-ID/near-dedup. **Overlap warning:** mbaza
  aggregates news (Kigali Today, Igihe), religious/cultural sites, **Wikipedia**, and
  govt/legal PDFs — so it overlaps our Wikipedia + the MADLAD web crawl. Cross-source
  near-dedup in Phase 2 is essential; raw token counts overstate unique content.

### MADLAD-400 — Kinyarwanda (rw), clean split
- **URL:** https://huggingface.co/datasets/allenai/MADLAD-400
  shard: https://huggingface.co/datasets/allenai/MADLAD-400/resolve/main/data/rw/rw_clean_0000.jsonl.gz
  (verified 2026-06-30)
- **License:** **CC-BY-4.0** — commercial use OK **with attribution**. Underlying
  text is Common Crawl; original page copyrights still apply. Cite Kudugunta et al.,
  *MADLAD-400* (2023).
- **Date downloaded:** 2026-06-30
- **Raw size:** 277.7 MB compressed (.jsonl.gz, one JSON/{"text":...} per line);
  ~1 GB decompressed. (Noisy split `rw_noisy_0000.jsonl.gz` = 736.5 MB — not used.)
- **Local path:** `data/sources/madlad/raw/rw_clean_0000.jsonl.gz` (raw, gitignored)
  → `data/sources/madlad/madlad_rw_clean.txt` (cleaned)
- **Reader script:** `data/prepare_madlad.py`
- **Cleaning applied:** stream-decompress gz → json per line → unescape literal
  `\n`/`\t`/`\r` (MADLAD stores in-doc breaks as escaped `\n`) → split to lines →
  sentence-split → drop web boilerplate (pipe/©/URL lines, markup starts,
  low-letter-ratio, <12 chars) → **hash-based exact-line dedup** → one sentence per
  line, UTF-8, n'/by' elisions preserved.
- **Output (full clean split, uncapped):** **226,466 docs → 5,424,400 unique
  sentences, ≈155M tokens, 734.5 MB.** (`--max-out-mb 0`, `--split clean`.)
  The **noisy** split (`--split noisy`, 737 MB compressed, ~3 GB text) is available
  for even more volume but is much lower quality — defer to after Phase 2 dedup.
- **Notes:** Chosen as the "volume" web-crawl source after OSCAR (gated/suspended on
  HF) and CC-100 (no Kinyarwanda split — confirmed 404) both proved unusable. Bypasses
  HF `datasets` (v5 dropped loading-script support; MADLAD ships a script). Still
  web-noisy (site names, dates, mixed-language snippets) → Phase 2 must language-ID
  filter + near-dedup hard.

### Kinyarwanda Wikipedia (rwwiki)
- **URL:** https://dumps.wikimedia.org/rwwiki/20260601/rwwiki-20260601-pages-articles.xml.bz2
  (dump index https://dumps.wikimedia.org/rwwiki/ ; verified 2026-06-30, latest = 20260601)
- **License:** CC BY-SA 4.0 + GFDL (text). Commercial use permitted **with
  attribution**; share-alike applies. Verified at https://dumps.wikimedia.org/legal.html
  (2026-06-30). Caveat: dumps may contain not-yet-removed copyright infringements.
- **Date downloaded:** 2026-06-30
- **Raw size:** 12.2 MB compressed (.xml.bz2); 12,006 articles (ns 0, redirects skipped)
- **Local path:** `data/sources/wikipedia/raw/` (raw, gitignored) →
  `data/sources/wikipedia/wikipedia_rw.txt` (cleaned)
- **Reader script:** `data/prepare_wikipedia.py`
- **Cleaning applied:** stream-parse bz2 MediaWiki XML (stdlib iterparse) → article
  namespace only, skip redirects → strip refs/comments/wikitables → `mwparserfromhell`
  `strip_code` → drop residual markup (image px sizes, thumb keyword, pipe/table-row
  lines, low-letter-ratio lines) → sentence-split → exact-line dedup → one sentence
  per line, UTF-8, whitespace normalized, n'/by' elisions preserved.
- **Output:** **149,845 unique sentences, ~2.23M words, ≈3.47M tokens (@1.56 t/w), 16.0 MB**
- **Notes:** Re-run any time with `python data/prepare_wikipedia.py [--date YYYYMMDD]`.
  Console may show `?`/`�` for curly apostrophes — that is a Windows terminal display
  artifact only; the file is valid UTF-8 (verified). Finer noise (English snippets,
  near-dupes) is left for Phase 2 cleaning.

### MasakhaNER (Kinyarwanda) — already used for the demo tokenizer
- **URL:** https://github.com/masakhane-io/masakhane-ner  *(to re-verify before reuse)*
- **License:** to verify (Masakhane datasets are generally open; confirm exact terms)
- **Date downloaded:** pre-existing (bootstrapped before this log)
- **Raw size:** demo corpus = 2,844 sentences / ~92k tokens
- **Local path:** reconstructed into `data/kinyarwanda_corpus.txt`
- **Reader script:** `data/prepare_corpus.py`
- **Cleaning applied:** one sentence per line, UTF-8
- **Notes:** Demo corpus only — proves the pipeline, far too small to train a
  useful model. The trained tokenizer in `tokenizer/kinyarwanda_bpe/` was built
  from this text.

<!-- Phase 1 targets to verify & add (CLAUDE.md §8):
     Kinyarwanda Wikipedia dump, KINNEWS/KIRNEWS, MasakhaNEWS,
     OSCAR rw, CC-100 rw, Kinyarwanda Bible, Digital Umuganda, FLORES/TICO-19. -->
