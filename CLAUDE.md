# CLAUDE.md — Kinyarwanda LLM (Rwanda's own language model)

> This file is the operating manual for **Claude Code** working on this repository.
> Read it fully before doing anything. Re-read the relevant section at the start of
> every work session. When you finish a task, update `PROGRESS.md` (create it if it
> does not exist) with what you did, what worked, and what is next.

---

## 0. How Claude Code should use this file

1. **Always start by reading** this file and `PROGRESS.md`, then run `python scripts/validate_pipeline.py` to confirm the project is in a healthy state.
2. **Work in phases** (Section 7). Do not skip ahead. Each phase has a *Definition of Done*; do not move on until it is met.
3. **Research before guessing.** This is a low-resource language and a fast-moving field. Whenever you are unsure about a dataset URL, a license, a library API, a current free-GPU quota, or a best practice, **use web search / fetch real documentation** (see Section 10). Never invent dataset links, license terms, or benchmark numbers.
4. **Prefer free, open resources** for both data and compute (Section 8). Cost should be near zero. If a resource needs payment, find a free alternative or flag it for the human.
5. **Commit small, test often.** After each meaningful change: run the relevant script, then `git add -A && git commit -m "..."`.
6. **Keep a human in the loop** for: licensing decisions, anything involving people's data/privacy, evaluation by native speakers, and final model release.

---

## 1. Mission

**Build an open, useful large language model for Kinyarwanda, so Rwandans have a capable model in their own language.** It must be reproducible, built from free/open resources, and respectful of data licenses and the people the data comes from.

First concrete target (decide with the human if not already decided):
- **v0** — a small Kinyarwanda *base* (text-completion) model that demonstrably learns the language.
- **v1** — an *instruction-tuned* Kinyarwanda assistant that can follow simple instructions and answer questions.

Keep every artifact reproducible: scripts > notebooks, fixed seeds, logged configs.

---

## 2. Core technical insight (why this project is shaped the way it is)

Kinyarwanda is a **Bantu, agglutinative** language: a single word encodes subject/object markers, tense, aspect, negation, and a stem. Example: `ntibazabikora` ≈ "they will not do it."

The transformer architecture is **language-agnostic** — the same math models English or Kinyarwanda. What breaks for Kinyarwanda is the **tokenizer** and the **data**:

- An English-trained tokenizer (e.g. GPT-2's) shatters Kinyarwanda words into meaningless fragments, wasting context and hurting learning.
- Peer-reviewed work (**KinyaBERT**, Nzeyimana & Niyongabo Rubungo, ACL 2022) shows tokenization is *the* bottleneck for Kinyarwanda NLP, and that morphology-aware tokenization beats plain BPE.

**Therefore the highest-leverage work is (a) a Kinyarwanda-trained tokenizer and (b) a large, clean Kinyarwanda corpus.** The model code is the easy part and already exists here.

---

## 3. Current state — what already exists (verified)

This project was bootstrapped by adapting **Sebastian Raschka's _Build a Large Language Model (From Scratch)_** (https://github.com/rasbt/LLMs-from-scratch). The following is **done and tested**:

- **A Kinyarwanda BPE tokenizer**, trained on authentic Kinyarwanda text (reconstructed from the MasakhaNER Kinyarwanda corpus). Files in `tokenizer/kinyarwanda_bpe/`.
- **A measured efficiency win** over GPT-2's tokenizer on the same Kinyarwanda text (run `python tokenizer/compare_tokenizers.py` to reproduce):

  | Tokenizer | tokens/word | chars/token |
  |---|---|---|
  | GPT-2 (English BPE, book default) | 3.03 | 2.33 |
  | **Kinyarwanda BPE (ours)** | **1.56** | **4.53** |

  ≈**1.94× more efficient (49% fewer tokens/word)**. Example: `abanyarwanda` ("Rwandans") = 1 token for us vs 5 fragments for GPT-2; negation prefix `nti-` preserved.
- **The GPT model** (`model/gpt.py`), faithful to the book, wired to the Kinyarwanda tokenizer, with temperature/top-k sampling.
- **A data pipeline** (`data/prepare_corpus.py`) that turns CoNLL/text sources into one-sentence-per-line corpora, plus a demo corpus `data/kinyarwanda_corpus.txt` (2,844 sentences, ~92k tokens).
- **Training & generation scripts** (`scripts/train.py`, `scripts/generate.py`) adapted from the book's chapter 5.
- **A no-dependency validator** (`scripts/validate_pipeline.py`) confirming tokenizer round-trip + correct next-token offset. Reports the model as ~16.9M params for `RWANDA_TINY`.

**Known limitation:** the demo corpus only proves the pipeline. It is far too small to train a useful model. The training loop has **not** been run end-to-end yet (needs PyTorch + a GPU). The job now is to scale data → tokenizer → pretraining → finetuning.

---

## 4. Repository map

```
kinyarwanda-llm/
├── CLAUDE.md                    # this file
├── README.md                   # human-facing overview + roadmap
├── PROGRESS.md                 # (you maintain this) running log of work
├── data/
│   ├── prepare_corpus.py        # ingest/clean text -> one sentence per line
│   ├── kinyarwanda_corpus.txt   # demo corpus (replace/extend with real data)
│   └── sources/                 # (you create) raw downloaded corpora, one dir per source
├── tokenizer/
│   ├── train_tokenizer.py       # train byte-level BPE
│   ├── compare_tokenizers.py    # benchmark vs GPT-2
│   └── kinyarwanda_bpe/         # trained tokenizer artifacts
├── model/
│   ├── gpt.py                   # GPT model + sampling (from the book)
│   ├── config.py                # RWANDA_TINY / RWANDA_SMALL + train hparams
│   └── data_loader.py           # tokenizer wrapper + dataset/dataloader
├── scripts/
│   ├── train.py                 # pretraining loop
│   ├── generate.py              # inference
│   └── validate_pipeline.py     # no-torch correctness check
└── eval/                        # (you create) evaluation scripts + results
```

---

## 5. Environment & setup

- **Python** 3.10+.
- Core deps already used: `tokenizers` (0.23.x), and the **bundled offline GPT-2 BPE** from the book repo at `LLMs-from-scratch/ch02/02_bonus_bytepair-encoder/` (used by `compare_tokenizers.py`). If that path is absent, `git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git` next to this project.
- For training you need **PyTorch**. Locally: `pip install torch`. On free GPU (Colab/Kaggle) PyTorch is preinstalled.
- Create a `requirements.txt` early: `tokenizers`, `torch`, `numpy`, `tqdm`, `datasets`, `regex`, and add tools as you introduce them. Pin versions.
- **Reproducibility:** fix seeds (`torch.manual_seed`, `random.seed`, `numpy`), log the exact config dict with every run, and save it alongside checkpoints.

---

## 6. Conventions & guardrails

- **Scripts over notebooks** for anything that must reproduce. A notebook is fine for exploration; port the result into a script.
- **One sentence/document per line, UTF-8**, for all corpora. Normalize whitespace. Preserve Kinyarwanda orthography (including apostrophes in elisions like `n'`, `by'`).
- **Never commit large data or model weights to git.** Use `.gitignore` for `data/sources/`, `checkpoints/`, `*.pt`. Keep scripts + small configs in git.
- **Track data provenance.** For every source, record in `data/sources/SOURCES.md`: name, URL, license, date downloaded, size, and any cleaning applied.
- **Don't fabricate.** No invented URLs, license claims, metrics, or Kinyarwanda text. If unsure, research or ask.
- **Respect licenses** (Section 12). Some news corpora are non-commercial only.

---

## 7. The build plan (phases)

### Phase 0 — Verify & set up *(do first)*
- Run `python scripts/validate_pipeline.py`; confirm it passes.
- Create `requirements.txt`, `.gitignore`, `PROGRESS.md`, `data/sources/SOURCES.md`.
- **DoD:** clean repo, validator passes, logs/structure in place.

### Phase 1 — Data acquisition *(the hard 80%)*
Goal: assemble **as much clean Kinyarwanda text as possible** — aim for hundreds of MB to a few GB (KinyaBERT used ~2.4 GB). Pull from the free sources in Section 8. Write a small reader per source that emits one-sentence/doc-per-line text into `data/sources/<name>/`.
- **DoD:** ≥ several hundred MB of raw Kinyarwanda text, each source logged in `SOURCES.md` with license verified.

### Phase 2 — Corpus cleaning & dedup
- Language-ID filter (keep Kinyarwanda; drop English/French/other). Research a current method (e.g. fastText `lid` model, or a simple stopword/charset heuristic using the KINNEWS Kinyarwanda stopword list).
- Normalize unicode, strip boilerplate/markup, fix encoding issues.
- **Deduplicate** (exact + near-duplicate; research MinHash/`datasketch` or `text-dedup`). Duplication badly hurts LLM training.
- Produce a single shuffled `data/kinyarwanda_corpus.txt` (+ held-out `val.txt`, `test.txt`).
- **DoD:** one clean, deduped, shuffled corpus with documented token count and a train/val/test split.

### Phase 3 — Tokenizer (scale up)
- Retrain the tokenizer on the full corpus with a larger vocab (try **16k–50k**; sweep and pick by fertility on held-out text).
- Re-run `compare_tokenizers.py`; record fertility vs GPT-2 and vs vocab size.
- Update `model/config.py` `VOCAB_SIZE` to match.
- **DoD:** final tokenizer with documented fertility; config updated.

### Phase 4 — Pretraining
- Start with `RWANDA_TINY` to confirm the loss goes down and samples become Kinyarwanda-like, **then** scale to `RWANDA_SMALL` (or larger) on a free GPU.
- Add to `train.py` as needed: learning-rate warmup+cosine decay (book ch.5 has schedulers), gradient clipping, mixed precision (`torch.autocast`), checkpoint resume, and `tqdm`/logging. Research current best practice for small-LLM pretraining before scaling.
- Monitor **validation perplexity**; save best checkpoint.
- **DoD:** a base checkpoint whose val perplexity improves clearly over training and that generates coherent Kinyarwanda fragments.

### Phase 5 — Evaluation
- **Intrinsic:** held-out perplexity; tokenizer fertility.
- **Extrinsic / downstream:** finetune or probe on **KINNEWS** (news classification) and **MasakhaNER** (NER) and report accuracy/F1 against the published baselines. Research the current SOTA numbers to contextualize.
- **Human:** have native Kinyarwanda speakers rate fluency/factuality of samples. Numbers don't measure cultural fluency.
- **DoD:** an `eval/RESULTS.md` with intrinsic + at least one downstream metric + a human-eval summary.

### Phase 6 — Instruction tuning (the assistant, v1)
- Build/obtain Kinyarwanda instruction data: translate-and-verify existing open instruction sets, write seed instructions with native speakers, and/or research open multilingual instruction datasets that include Kinyarwanda. **Verify translations with native speakers; do not ship machine-translated instructions blindly.**
- Use the book's **chapter 7 (instruction finetuning)** and **chapter 6 (classification finetuning)** as the template.
- Optionally add **LoRA** (book appendix E) for cheap finetuning on free GPUs.
- **DoD:** an instruction-tuned checkpoint that follows simple Kinyarwanda instructions, with examples in `eval/`.

### Phase 7 — Serving & sharing
- A minimal CLI/`generate.py` is already here; add a small API or a Gradio/Streamlit demo.
- Publish openly (e.g. a model card + weights on a free host) **with correct licenses and attribution**, so Rwandans and researchers can use it. Write a clear **model card** (intended use, training data, limitations, biases, eval).
- **DoD:** a runnable demo + a model card; release approved by the human.

### Phase 8 — Research upgrade: morphology-aware tokenization
- Investigate replacing/augmenting plain BPE with a **morphology-aware** approach (the KinyaBERT direction: a morphological analyzer + two-tier encoding). Research existing Kinyarwanda morphological analyzers and tooling first.
- **DoD:** a documented experiment comparing morphology-aware vs BPE on fertility and at least one downstream task.

---

## 8. Free resources to use (and research further)

> **Directive:** treat this list as a *starting point*. **Before downloading, verify each link, its current availability, and its license by fetching the real page.** Find additional Kinyarwanda sources beyond this list — search actively. Log everything in `data/sources/SOURCES.md`.

**Text data (free / open):**
- **Kinyarwanda Wikipedia** dump (`rw.wikipedia.org`) — via Wikimedia dumps; clean with a wiki extractor.
- **KINNEWS / KIRNEWS** — 21k+ Kinyarwanda news articles (Niyongabo Rubungo et al., COLING 2020). Repo: `github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus` (data on the linked Google Drive). Includes a **Kinyarwanda stopword list** useful for language-ID/cleaning.
- **MasakhaNER / MasakhaNEWS** (Masakhane) — Kinyarwanda splits hosted on GitHub / Hugging Face `datasets`.
- **OSCAR** and **CC-100** — `rw` (Kinyarwanda) web-crawl splits (via Hugging Face `datasets`); large but noisy → clean hard.
- **mC4** `rw` split (if available) — large web text.
- **Kinyarwanda Bible** translations and other public-domain religious texts — large, clean parallel/monolingual text.
- **Digital Umuganda** open Kinyarwanda corpora (`github.com/Digital-Umuganda`).
- **FLORES / TICO-19** — small but clean parallel sets including Kinyarwanda (good for eval and MT-style finetuning).
- Research more: Rwandan news sites, government open data, OPUS parallel corpora, and any new African-NLP releases.

**Free compute (research current quotas before relying on them):**
- **Google Colab** (free tier GPU), **Kaggle Notebooks** (weekly free GPU hours), **Hugging Face Spaces**, **Lightning AI** / **Paperspace** free tiers, and any current academic/cloud credit programs. Prefer whichever currently gives the most free GPU-hours for your model size.

**Free tooling:**
- `tokenizers` (BPE/Unigram), Hugging Face `datasets` & `transformers`, `fasttext` (language-ID), `datasketch`/`text-dedup` (dedup), `accelerate` (multi-device), `gradio`/`streamlit` (demo). The book repo for architecture reference.

---

## 9. Prompt library (copy-paste task prompts)

These are concrete prompts a human can give Claude Code, or that Claude Code can self-issue. Each assumes this `CLAUDE.md` is loaded.

**Setup**
> "Read CLAUDE.md and PROGRESS.md. Run scripts/validate_pipeline.py and report status. Then create requirements.txt (pinned), .gitignore (ignore data/sources, checkpoints, *.pt), PROGRESS.md, and data/sources/SOURCES.md with empty templates. Commit."

**Data acquisition**
> "Phase 1. Research and verify free Kinyarwanda text sources from CLAUDE.md Section 8 by fetching each page and confirming the license. For each viable source, write data/prepare_<source>.py that downloads it and emits one-sentence-per-line UTF-8 into data/sources/<source>/. Log name, URL, license, date, and size in SOURCES.md. Start with Kinyarwanda Wikipedia and MasakhaNEWS. Do not invent URLs — if a link 404s, search for the current one."

**Cleaning & dedup**
> "Phase 2. Write data/clean_corpus.py that: language-ID filters to Kinyarwanda (research and use a current method), normalizes unicode/whitespace, removes boilerplate, and deduplicates (research MinHash-based near-dedup). Output data/kinyarwanda_corpus.txt plus val.txt/test.txt splits. Print token counts before/after each step. Commit and update PROGRESS.md."

**Tokenizer scale-up**
> "Phase 3. Retrain the tokenizer on the full corpus for vocab sizes [16000, 32000, 50000]. For each, report fertility (tokens/word) on val.txt via compare_tokenizers.py. Pick the best trade-off, save it to tokenizer/kinyarwanda_bpe/, update VOCAB_SIZE in model/config.py, and write the comparison to eval/tokenizer_sweep.md."

**Pretraining**
> "Phase 4. First confirm RWANDA_TINY trains and loss decreases on CPU/GPU for a few hundred steps and samples look Kinyarwanda-like. Then add LR warmup+cosine decay, gradient clipping, mixed precision, checkpoint resume, and tqdm logging to scripts/train.py. Research current best practice for small-LLM pretraining hyperparameters before scaling. Then train RWANDA_SMALL on the available free GPU, logging val perplexity, and save the best checkpoint."

**Evaluation**
> "Phase 5. Write eval/ scripts to (a) compute held-out perplexity and (b) finetune/probe the base model on KINNEWS classification and MasakhaNER, reporting accuracy/F1 vs the published baselines (research current numbers). Summarize in eval/RESULTS.md. Prepare 20 generation samples for native-speaker review."

**Instruction tuning**
> "Phase 6. Research open instruction datasets that include or can be adapted to Kinyarwanda. Build a small, native-speaker-verified Kinyarwanda instruction set. Implement instruction finetuning following LLMs-from-scratch chapter 7 (optionally LoRA from appendix E). Save the chat model and add example dialogues to eval/."

**Serving**
> "Phase 7. Build a minimal Gradio demo around generate.py. Write a model card (intended use, data, license, limitations, biases, eval results). Propose a free hosting plan for weights + demo and list the exact license/attribution required by each training source."

---

## 10. Research protocol (how to handle what you don't know)

This field and these resources change. **Do not rely on possibly-stale memory for anything external.** When you need a fact about a dataset, library API, license, benchmark number, current free-GPU quota, or best practice:

1. **Search**, then **fetch the actual page/docs** and read it. Prefer primary sources (official repos, papers, dataset cards, library docs).
2. **Verify before use:** confirm a download link resolves and the license permits your use *before* writing code around it.
3. **Cross-check** surprising or important numbers across two sources.
4. **Record** what you found (source + date) in `PROGRESS.md` or `SOURCES.md`, so the next session doesn't redo it.
5. **If a resource is paywalled or unavailable, find a free alternative** or flag it for the human — do not silently substitute or fabricate.

---

## 11. Definition of done (project milestones)

- **M1:** Clean, deduped, license-cleared Kinyarwanda corpus (token count documented) + final tokenizer with measured fertility.
- **M2:** A base model checkpoint with clearly improving val perplexity that generates coherent Kinyarwanda.
- **M3:** Downstream eval (KINNEWS and/or MasakhaNER) + native-speaker fluency review.
- **M4:** Instruction-tuned Kinyarwanda assistant (v1) with a demo and a model card.
- **M5:** Open release with correct licenses and attribution.

---

## 12. Risks, ethics & licensing

- **Licensing:** Each corpus has its own terms; some news data is **non-commercial only**. Verify and record every license. Don't mix incompatible licenses into a release without checking.
- **Privacy:** Web/news text can contain personal data. Avoid memorizing/echoing private info; dedup and filter.
- **Bias & safety:** News-heavy data skews topic/tone. Document limitations in the model card. Have native speakers check for harmful or biased outputs.
- **Community respect:** This is *for* Rwandans — involve Kinyarwanda speakers in data creation, evaluation, and release decisions. Credit the open datasets and the Masakhane / KINNEWS / KinyaBERT authors whose work this stands on.
- **Reproducibility:** Anyone should be able to rebuild from these scripts + documented sources.

---

## 13. References (verify/expand as you go)

- Raschka, S. *Build a Large Language Model (From Scratch)* — https://github.com/rasbt/LLMs-from-scratch (architecture, training loop, instruction tuning, LoRA).
- Nzeyimana & Niyongabo Rubungo (2022). *KinyaBERT: a Morphology-aware Kinyarwanda Language Model.* ACL 2022 — motivation for morphology-aware tokenization.
- Niyongabo Rubungo et al. (2020). *KINNEWS and KIRNEWS.* COLING 2020 — news corpus + stopwords + classification benchmark.
- Masakhane: **MasakhaNER**, **MasakhaNEWS** — Kinyarwanda NER/news data and African-NLP community resources.

---

*Remember: the model code is the easy part — it already works. The mission is won or lost on **data quality** and **tokenization**. Spend your effort there, use free/open resources, research what you don't know, verify everything, and keep Rwandans in the loop. Let's give Rwanda its model.*
