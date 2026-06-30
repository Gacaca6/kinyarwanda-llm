# PROGRESS — Kinyarwanda LLM

Running log of work (CLAUDE.md §0). Newest entries on top. Update at the end of
every working session with: what was done, what worked, what's next.

---

## 2026-06-30 — Phase 1: Data maximization — ≈193M tokens across 3 sources

Decision (user): "maximize data first" before training. Executed:

- **Uncapped MADLAD clean** (`--max-out-mb 0`): 226,466 docs → 5,424,400 sentences →
  ~156M tokens / 740 MB (was 500 MB capped). Generalized `prepare_madlad.py` with
  `--split {clean,noisy}`.
- **Added mbaza NLP v01.0** (`data/prepare_mbaza.py`, pyarrow): 78,733 docs →
  966,610 sentences → ~34M tokens / 159 MB. CC BY 4.0. (v01.1 gated → used open v01.0.)
- **Ruled out (verified, flagged not faked):**
  - MasakhaNEWS — no Kinyarwanda config (has Rundi, not kin) + CC-BY-**NC**.
  - mbaza v01.1 — gated (HF login).
  - Kinyarwanda Bible — not in public-domain BibleNLP/ebible (1,080 translations, 0 kin);
    Bibiliya Yera is © Bible Society of Rwanda. No clean bulk download found.

**Corpus now (raw, pre-Phase-2):**
| Source | sentences | ~tokens | size |
|---|---|---|---|
| Wikipedia rw | 149,845 | ~3.5M | 16 MB |
| MADLAD-400 rw clean (full) | 5,424,400 | ~156M | 740 MB |
| mbaza rw v01.0 | 966,610 | ~34M | 159 MB |
| **Total** | **6,540,855** | **~193M** | **~916 MB** |

**Caveat:** heavy cross-source overlap (mbaza contains Wikipedia + news; MADLAD crawl
includes news sites). Unique tokens after Phase 2 near-dedup will be materially lower.

**Remaining volume lever (not done):** MADLAD **noisy** split (`--split noisy`, 737 MB
compressed, ~3 GB text). Held back: low quality + in-reader hash-dedup set would need
several GB RAM at that line count. Better after Phase 2 has scalable (MinHash) dedup.

**Next:** Phase 2 — language-ID filter (drop non-rw), normalize, cross-source
near-dedup, then shuffled train/val/test split with documented token counts.

---

## 2026-06-30 — Phase 1: Data acquisition — DoD MET (≈516 MB across 2 sources)

**Done — MADLAD-400 rw (clean split) ingested (the "volume" source)**
- The chosen targets failed verification (CLAUDE.md §10, flagged not faked):
  - **OSCAR** — gated + access suspended on HF. Unusable without manual approval.
  - **CC-100** — has **no Kinyarwanda split** (rw.txt.xz 404s; confirmed absent from index).
- Verified substitute: **MADLAD-400** (`allenai/MADLAD-400`) — ungated, **CC-BY-4.0**
  (attribution), includes `rw`. Shard `data/rw/rw_clean_0000.jsonl.gz` = 277.7 MB
  compressed (~1 GB text).
- Wrote `data/prepare_madlad.py`. Pulls the raw .jsonl.gz shard directly via URL
  (HF `datasets` v5 dropped loading-script support; MADLAD ships a script). Streams
  json → unescapes literal `\n` → sentence-splits → drops web boilerplate →
  hash-based exact dedup (bounded memory). Output cap default 500 MB (`--max-out-mb`).
- Ran it (cap 500 MB):
  **153,171 docs → 3,697,960 unique sentences, ~68.2M words, ≈106M tokens, 500.0 MB.**
  Output: `data/sources/madlad/madlad_rw_clean.txt`.
- Logged full provenance in `data/sources/SOURCES.md`. Added `huggingface_hub` to reqs.

**Corpus so far (raw, pre-Phase-2):**
| Source | sentences | ~tokens | size |
|---|---|---|---|
| Wikipedia rw | 149,845 | ≈3.47M | 16 MB |
| MADLAD-400 rw clean (capped) | 3,697,960 | ≈106M | 500 MB |
| **Total** | **3.85M** | **≈110M** | **≈516 MB** |

**Phase 1 DoD** ("≥ several hundred MB, each source license-verified & logged"): **MET.**

**Verified facts**
- MADLAD encodes in-doc newlines as the literal 2-char `\n` — must unescape post-json.
- `datasets` 5.0.0 refuses script-based datasets ("Dataset scripts are no longer
  supported") — pull MADLAD shards by URL instead.
- MADLAD clean is still web-noisy (site names, dates, some non-rw) → Phase 2 job.

**Next options**
- (a) Move to **Phase 2** (clean/dedup/lang-ID across both sources → train/val/test).
- (b) Or add more Phase 1 sources first (mbazaNLP rw set, MasakhaNEWS, Bible) — more
  data never hurts, but the DoD is already met so Phase 2 is the higher-value next step.

---

## 2026-06-30 — Phase 1 (started): Data acquisition — source 1

**Done — Kinyarwanda Wikipedia ingested**
- Verified (fetched real pages, CLAUDE.md §10):
  - Dump index `https://dumps.wikimedia.org/rwwiki/` → latest **20260601**;
    file `rwwiki-20260601-pages-articles.xml.bz2` (12.2 MB compressed).
  - License **CC BY-SA 4.0 + GFDL**, commercial OK **with attribution**
    (https://dumps.wikimedia.org/legal.html).
- Wrote `data/prepare_wikipedia.py` (deps: `mwparserfromhell==0.7.2`, stdlib
  bz2/xml/urllib). Streams the bz2 XML, keeps article namespace, skips redirects,
  strips wiki markup, emits one clean sentence per line, exact-line dedup.
- Ran it on the full dump:
  **12,006 articles → 149,845 unique sentences, ~2.23M words, ≈3.47M tokens, 16.0 MB.**
  (~38× the demo corpus's 92k tokens.) Output: `data/sources/wikipedia/wikipedia_rw.txt`.
- Logged full provenance in `data/sources/SOURCES.md`.
- Added `mwparserfromhell>=0.6` to `requirements.txt`.

**Verified facts**
- Wikimedia dumps return HTTP 403 without a descriptive User-Agent — script sets one.
- File is valid UTF-8 with n'/by' elisions preserved; `�` in the Windows console is
  a display artifact only.

**Phase 1 status:** NOT done yet. Wikipedia is source 1. Phase 1 DoD wants several
hundred MB; need more sources next.

**Next (Phase 1, continue):**
- MasakhaNEWS (Kinyarwanda) via Hugging Face `datasets` — verify license, write
  `data/prepare_masakhanews.py`.
- Then KINNEWS/KIRNEWS, then OSCAR rw / CC-100 rw (large, noisy), Kinyarwanda Bible.
- Heavy cleaning + lang-ID + near-dedup across all sources is **Phase 2**, not now.

---

## 2026-06-30 — Phase 0: Verify & set up

**Done**
- Extracted the project tree from `kinyarwanda-llm.zip` into `kinyarwanda-llm/`
  (this directory is now the repo root; placed the operating manual here as
  `CLAUDE.md`).
- Environment: Windows 11, Python 3.13.2. Installed `tokenizers==0.23.1`.
- Ran `python scripts/validate_pipeline.py` — **PASS**:
  - tokenizer round-trip OK (`'Abanyarwanda bishimiye uburezi.'`)
  - corpus = **92,152 tokens** → **359 training pairs** (256 tokens each)
  - next-token offset check PASS
  - analytic model size **16.9M params** (RWANDA_TINY)
- Created Phase 0 scaffolding:
  - `requirements.txt` (pinned `tokenizers`; floors for torch/numpy/tqdm/datasets/regex)
  - `.gitignore` (ignores `data/sources/`, `checkpoints/`, `*.pt`, caches)
  - `data/sources/SOURCES.md` (provenance template + MasakhaNER demo entry)
  - `PROGRESS.md` (this file)

**Verified facts (so the next session doesn't redo them)**
- Validator needs only `tokenizers` (no torch) — good for CPU-only checks.
- Demo corpus is MasakhaNER-derived, ~92k tokens — proves pipeline only.

**Phase 0 Definition of Done** — clean repo, validator passes, logs/structure in
place: **MET** (pending optional `git init` + initial commit).

**Next — Phase 1: Data acquisition (the hard 80%)**
- Goal: assemble hundreds of MB–GBs of clean Kinyarwanda text.
- First targets (verify URL + license by fetching the page first, CLAUDE.md §8/§10):
  Kinyarwanda Wikipedia dump, then MasakhaNEWS, then KINNEWS/KIRNEWS.
- For each viable source: write `data/prepare_<source>.py` emitting one
  sentence/doc per line UTF-8 into `data/sources/<source>/`; log in `SOURCES.md`.

**Open items / for the human**
- Confirm whether to `git init` here and the intended remote (if any).
- Loose duplicate copies of `gpt.py`, `train.py`, `compare_tokenizers.py` and the
  original `kinyarwanda-llm.zip` remain in the parent `Rwandan LLM/` folder —
  redundant with the extracted tree; can be removed on your say-so.
