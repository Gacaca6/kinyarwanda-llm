# Kinyarwanda LLM — a foundation for a model "for Rwandans"

This project adapts **[*Build a Large Language Model (From Scratch)*](https://github.com/rasbt/LLMs-from-scratch)** by Sebastian Raschka into a starting point for a Kinyarwanda language model. The book gives a complete, language-agnostic GPT implementation (tokenizer → attention → transformer → pretraining → finetuning). This repo keeps that architecture and changes the two things that actually make a model work *for Rwandans*: **the tokenizer** and **the data**.

## The core insight

A transformer doesn't care what language it models — the same math works for English or Kinyarwanda. What breaks for Kinyarwanda is everything *around* the transformer, above all **tokenization**.

Kinyarwanda is a Bantu, **agglutinative** language: one word packs subject/object markers, tense, aspect, negation, and a stem. `ntibazabikora` ≈ "they will not do it." The book ships with GPT‑2's tokenizer, whose merges were learned on English, so it shatters Kinyarwanda words into meaningless fragments. This wastes context and makes learning harder. Peer‑reviewed work on the language — **KinyaBERT** (Nzeyimana & Niyongabo Rubungo, ACL 2022) — shows tokenization is *the* bottleneck for Kinyarwanda NLP.

So the first, highest‑leverage adaptation is training the tokenizer on Kinyarwanda itself.

## What's been built and proven here

A real Kinyarwanda BPE tokenizer was trained on authentic text (reconstructed from the MasakhaNER Kinyarwanda corpus) and measured against the book's default GPT‑2 tokenizer **on the same Kinyarwanda text**:

| Tokenizer | tokens / word | chars / token |
|---|---|---|
| GPT‑2 (English BPE, book default) | 3.03 | 2.33 |
| **Kinyarwanda BPE (ours)** | **1.56** | **4.53** |

**≈1.94× more efficient — 49% fewer tokens per word.** And the splits are linguistically sensible:

```
abanyarwanda   GPT-2 (5): ['ab','any','ar','w','anda']   ours (1): ['abanyarwanda']
ntibazabikora  GPT-2 (6): ['nt','ib','az','ab','ik','ora'] ours (3): ['nti','baza','bikora']
icyorezo       GPT-2 (3): ['icy','ore','zo']              ours (1): ['icyorezo']
```

"Abanyarwanda" (*Rwandans*) collapses from 5 fragments to a single token, and the negation prefix `nti-` is preserved as a unit.

## Project layout

```
kinyarwanda-llm/
├── data/
│   ├── prepare_corpus.py        # ingest + clean text -> one sentence per line
│   └── kinyarwanda_corpus.txt   # 2,844 sentences (demo corpus, ~92k tokens)
├── tokenizer/
│   ├── train_tokenizer.py       # train Kinyarwanda byte-level BPE
│   ├── compare_tokenizers.py    # efficiency benchmark vs GPT-2 (table above)
│   └── kinyarwanda_bpe/         # trained tokenizer (vocab.json, merges.txt, tokenizer.json)
├── model/
│   ├── gpt.py                   # GPT model (from the book) + sampling
│   ├── config.py               # RWANDA_TINY / RWANDA_SMALL sizes + train hparams
│   └── data_loader.py          # Kinyarwanda tokenizer wrapper + dataset/dataloader
└── scripts/
    ├── train.py                 # pretraining loop (from book ch.5)
    ├── generate.py              # load checkpoint + generate Kinyarwanda
    └── validate_pipeline.py     # no-torch correctness check (tokenizer + chunking)
```

## How to run

```bash
# 1. (optional) rebuild the corpus from CoNLL sources
python data/prepare_corpus.py <file1.txt> <file2.txt> data/kinyarwanda_corpus.txt

# 2. train the Kinyarwanda tokenizer
cd tokenizer && python train_tokenizer.py --vocab_size 8000 && cd ..

# 3. see the efficiency gain vs GPT-2
cd tokenizer && python compare_tokenizers.py && cd ..

# 4. pretrain (needs PyTorch + ideally a GPU)
pip install torch
python -m scripts.train

# 5. generate
python -m scripts.generate "Abanyarwanda"
```

`scripts/validate_pipeline.py` runs without PyTorch and confirms the tokenizer round‑trips and the training pairs are correctly offset.

## The honest part: this is a foundation, not a finished model

The demo corpus here (~92k tokens) is enough to **prove the pipeline**, not to train a useful model. A genuinely useful Kinyarwanda LLM is a resourced, multi‑month effort. The bottleneck is **data**, not code.

**Roadmap**

1. **Scale the corpus (the hard 80%).** Target hundreds of MB to a few GB of clean Kinyarwanda. KinyaBERT used ~2.4 GB. Real public sources:
   - Kinyarwanda Wikipedia dump (`rw.wikipedia.org`)
   - KINNEWS — 21k+ Kinyarwanda news articles (Niyongabo Rubungo et al., 2020)
   - MasakhaNER / MasakhaNEWS Kinyarwanda splits (Masakhane)
   - OSCAR and CC‑100 `rw` splits (web crawl)
   - Kinyarwanda Bible translations and other public-domain religious texts
   - Digital Umuganda's open Kinyarwanda corpora
   - Each source plugs into the same `prepare_corpus.py` pattern (one reader → one sentence per line).
2. **Deduplicate and clean** aggressively (near‑duplicate removal, language‑ID filtering, normalization). Data quality matters more than model size at this scale.
3. **Retrain the tokenizer** on the full corpus (vocab ~32k–50k once you have the data).
4. **Pretrain** `RWANDA_SMALL` on a GPU; scale `emb_dim`/`n_layers` with the corpus.
5. **Instruction‑finetune** for an assistant, using the book's chapter 7 (instruction tuning) and chapter 6 (classification) as the template; build Kinyarwanda instruction data.
6. **Upgrade the tokenizer (research direction).** Move from plain BPE toward a **morphology‑aware** tokenizer (the KinyaBERT idea: a morphological analyzer + two‑tier encoding) to capture Kinyarwanda's structure even better.

**Practical and ethical notes**
- Respect each data source's license; some news corpora are non‑commercial only.
- Involve native Kinyarwanda speakers in evaluation and instruction‑data creation — fertility numbers don't measure cultural fluency or safety.
- Plan compute realistically: even a small model needs a GPU for pretraining; a 16–30M‑param model is a reasonable first target on a single GPU.

## Credit

- Architecture, training loop, and the bundled GPT‑2 BPE encoder: **Sebastian Raschka, *Build a Large Language Model (From Scratch)***, https://github.com/rasbt/LLMs-from-scratch
- Kinyarwanda text used to train the demo tokenizer: **MasakhaNER** (Masakhane / Adelani et al.)
- Tokenization motivation: **KinyaBERT** (Nzeyimana & Niyongabo Rubungo, ACL 2022); **KINNEWS/KIRNEWS** (Niyongabo Rubungo et al., COLING 2020)
