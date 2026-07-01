# PROGRESS — Kinyarwanda LLM

Running log of work (CLAUDE.md §0). Newest entries on top. Update at the end of
every working session with: what was done, what worked, what's next.

---

## 2026-07-01 — Phase 4 (in progress): Pretraining — loop PROVEN, GPU run handed off

**The never-run training loop now runs end-to-end.**

- Added `scripts/prepare_tokens.py`: tokenizes the corpus into flat uint16 memmap
  bins (nanoGPT style — low RAM vs the book's per-chunk tensors). **Real token counts
  (32k tokenizer):** train.bin **173,854,839**, val.bin 1,779,443, test.bin 1,773,489
  → **~177.4M tokens** (matches the 1.56 word estimate almost exactly).
- Rewrote `scripts/train.py` for real pretraining: memmap random-batch sampling,
  AdamW (no decay on biases/norms), linear warmup → cosine decay, grad clipping,
  mixed precision (CUDA), val-loss→perplexity, best-checkpoint + `--resume`, tqdm,
  periodic sampling, `--model tiny|small`, argparse. Reads `data/processed/*.bin`.
- Installed PyTorch 2.12.1+cpu locally (no GPU on this machine).
- **CPU smoke test** (RWANDA_TINY, 150 steps, 1.46M-token slice):
  val_loss **10.50 → 7.64**, ppl **36,466 → 2,087**, falling steadily; sample already
  emits real Kinyarwanda (`mu Rwanda`, `mu gihe`, `uyu`, `ku`, `ko`). Loop confirmed.
- **GPU handoff** (no local GPU → real RWANDA_SMALL run on free cloud GPU):
  - `kaggle/` — Kaggle notebook + guide. **Blocked for this user:** Kaggle requires
    phone verification to unlock GPU and it wasn't working for them.
  - `colab/` (**primary path now**) — Google Colab needs no phone/card. Mounts Google
    Drive for the bins AND for persistent checkpoints (survives Colab's ~12h session
    limit; re-run cell to `--resume`). Same `train.py`. See `colab/README.md`.
  - Other phone-free fallbacks noted: SageMaker Studio Lab, Lightning AI, Paperspace.

**Status:** Phase 4 step 1 (prove loss drops + Kinyarwanda-like samples) **MET** on CPU.
Remaining: the real GPU pretraining run on Kaggle (RWANDA_SMALL, watch val perplexity)
→ produces the M2 base checkpoint. That run is the user's to launch.

**Verified facts**
- torch 2.12.1+cpu works here; `cuda=False`. AMP path is a no-op on CPU.
- RWANDA_TINY is 35.3M params at 32k vocab (embeddings dominate at tiny size).
- Data-limited regime: ~177M tokens vs Chinchilla-optimal ~2.5B for RWANDA_SMALL
  (124M) → train multiple epochs, save best by val ppl.

**Next:** run `kaggle/kaggle_train.py` on Kaggle GPU → M2 checkpoint → Phase 5 eval
(perplexity on test.bin, KINNEWS/MasakhaNER downstream, native-speaker review).

---

## 2026-06-30 — Phase 3: Tokenizer scale-up — DONE (DoD met)

Wrote `tokenizer/sweep_tokenizer.py`: trains byte-level BPE on the 178M-token
`train.txt` at vocab 16k/32k/50k, measures fertility on `val.txt`, writes
`eval/tokenizer_sweep.md`. Baseline = GPT-2 via `tiktoken`.

**Sweep (fertility = tokens/word on val.txt, lower better):**
| vocab | fertility | chars/tok | vs GPT-2 |
|---|---|---|---|
| GPT-2 (50257) | 3.07 | 2.33 | 1.00x |
| 16k | 1.59 | 4.49 | 1.93x |
| **32k (chosen)** | **1.47** | **4.85** | **2.08x** |
| 50k | 1.42 | 5.03 | 2.16x |

**Decision: 32k.** Gains diminish (16k→32k −7.5%, 32k→50k only −3.4%) while embedding
+ head cost grows linearly (24.6M/49.2M/76.8M params at emb_dim 768). 50k would make
embeddings ~half a RWANDA_SMALL model. At 178M tokens each 32k type is seen ~5,500x.

**Done:** promoted 32k tokenizer → `tokenizer/kinyarwanda_bpe/`; set
`VOCAB_SIZE=32000` in `model/config.py`; updated `validate_pipeline.py` CFG; **fixed
`compare_tokenizers.py`** (was hardcoded to a Linux `/home/claude` path → now tiktoken,
runs anywhere). Validator passes (round-trip OK; TINY now 35.3M params at 32k vocab).
`ntibazabikora` → `['nti','baza','bikora']`; `abanyarwanda`/`umunyarwanda` → 1 token.

**Phase 3 DoD** (final tokenizer with documented fertility; config updated): **MET.**
Added `tiktoken` to requirements. Sweep intermediates gitignored (`tokenizer/sweep/`).

**Milestone M1 (corpus + tokenizer) — essentially COMPLETE.**

**Next — Phase 4 (pretraining):** first confirm RWANDA_TINY loss drops + samples look
Kinyarwanda-like (CPU/GPU, a few hundred steps), then add LR warmup+cosine, grad
clipping, mixed precision, checkpoint resume, tqdm; train on a free GPU; track val
perplexity. Needs PyTorch (not yet installed). NB: `train.py` currently reads
`data/kinyarwanda_corpus.txt` — point it at `data/processed/train.txt` + use `val.txt`.

---

## 2026-06-30 — Phase 2: Cleaning, dedup & split — DONE (DoD met)

Built `data/clean_corpus.py`: combine all `data/sources/*/*.txt` → normalize →
language-filter → dedup → shuffle → train/val/test. Outputs to `data/processed/`
(gitignored, large).

**Language-ID — key finding:** fastText `lid.176` (via fast-langdetect) **cannot
detect Kinyarwanda** — on real rw it misfires to hr/id/sw/en at <0.3 confidence,
while nailing en/fr at >0.85. So (per CLAUDE.md's sanctioned fallback) we use the
**KINNEWS Kinyarwanda stopword list** (`data/lang_id/listed.txt`, 80 words) as a
*negative* filter: drop a line only when it's confidently foreign (>=2 EN/FR function
words AND 0 Kinyarwanda function words) + a non-Latin-script drop. On our (already
~95% rw) sources this keeps ~99% and precisely removes the foreign minority.

**Dedup:** collapse on a normalized key (casefold + letters/digits only + single
spaces), 128-bit blake2b. Removes exact dups, trivial variants, and cross-source
overlap (mbaza re-includes Wikipedia/news that MADLAD also crawled). Normalized-exact;
semantic/MinHash near-dedup noted as a future refinement.

**Results (`SEED=123`, split 98/1/1):**
| | lines | words | ~tokens | size |
|---|---|---|---|---|
| raw (3 sources) | 6,540,855 | 123,896,175 | — | — |
| after filter+dedup | 6,081,561 | 116,224,416 | ~181M | — |
| **train.txt** | 5,959,931 | 113,897,871 | **~177.7M** | 837 MB |
| val.txt | 60,815 | 1,165,183 | ~1.8M | 9 MB |
| test.txt | 60,815 | 1,161,362 | ~1.8M | 9 MB |

Removed 459,294 lines (7.0%: foreign/junk/duplicate). **Leakage check: val∩test = 0**;
train disjoint by construction (dedup precedes split). No train/test contamination.

**Phase 2 DoD** (one clean, deduped, shuffled corpus + documented token count +
train/val/test split): **MET.**

**Notes / decisions**
- Outputs go to `data/processed/` (gitignored), NOT `data/kinyarwanda_corpus.txt` —
  the "never commit large data" guardrail outranks the manual's example filename. The
  small demo `data/kinyarwanda_corpus.txt` stays as-is.
- `clean_corpus.py` reconfigures stdout to UTF-8 (Windows cp1252 consoles choke on
  non-ASCII otherwise).
- Token counts are word×1.56 estimates; real counts come after the Phase 3 retokenize.

**Next — Phase 3 (tokenizer scale-up):** retrain BPE on `data/processed/train.txt`
at vocab 16k/32k/50k, pick by fertility on `val.txt`, update `VOCAB_SIZE`.

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
