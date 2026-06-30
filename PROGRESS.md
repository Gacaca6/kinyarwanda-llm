# PROGRESS — Kinyarwanda LLM

Running log of work (CLAUDE.md §0). Newest entries on top. Update at the end of
every working session with: what was done, what worked, what's next.

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
