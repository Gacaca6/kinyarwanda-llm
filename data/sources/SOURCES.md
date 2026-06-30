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

## Sources logged

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
- **Output (capped at 500 MB):** **153,171 docs → 3,697,960 unique sentences,
  ~68.2M words, ≈106M tokens, 500.0 MB.** Cap is configurable: re-run with
  `--max-out-mb 0` to process the full ~1 GB.
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
