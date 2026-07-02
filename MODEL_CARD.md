# Model Card — Kinyarwanda LLM (base, v0)

## Model summary
A from-scratch GPT-style language model for **Kinyarwanda** (ISO 639-3: `kin`), a
Bantu, agglutinative language. It is a **base (text-completion) model**: it continues
Kinyarwanda text. It is **not** an instruction-following assistant and does not answer
questions or follow commands.

- **Architecture:** decoder-only Transformer (GPT), 12 layers, hidden dim 768, 12
  heads, context length 512. ~134.6M parameters.
- **Tokenizer:** byte-level BPE trained on Kinyarwanda, vocab 32,000. ~1.47
  tokens/word — **2.08× more token-efficient than GPT-2's English BPE** on Kinyarwanda.
- **Training data:** ~174M tokens of clean, deduplicated Kinyarwanda (see below).
- **Status:** v0 base model. Demonstrably learns the language; modest in size and
  data. Built on Sebastian Raschka's *Build a Large Language Model (From Scratch)*.

## Intended use
- Kinyarwanda **text completion / continuation**.
- A **base for fine-tuning**: it already reaches 87.4% on KINNEWS news classification
  after a lightweight fine-tune (see Evaluation), so it is a useful starting point for
  Kinyarwanda downstream tasks.
- Research and education on low-resource / African-language NLP.

### Out of scope / do not use for
- **Not** an assistant/chatbot — it will not reliably follow instructions.
- **Not** a source of facts — it can produce fluent but false statements
  (hallucinations) and repetition. Do not use for medical, legal, financial, or other
  high-stakes decisions.
- Not safety-tuned; it can reflect biases in web/news training text.

## How to use
```python
import torch
from model.gpt import GPTModel, generate
from model.data_loader import KinyaTokenizer

tok = KinyaTokenizer("tokenizer/kinyarwanda_bpe/tokenizer.json")
ck = torch.load("kinyarwanda_small_v0.pt", map_location="cpu")
model = GPTModel(ck["config"]); model.load_state_dict(ck["model_state_dict"]); model.eval()

ids = torch.tensor([tok.encode("U Rwanda")])
out = generate(model, ids, 60, ck["config"]["context_length"],
               temperature=0.8, top_k=40, eos_id=tok.eot_id)
print(tok.decode(out[0].tolist()))
```
Or run the demo: `pip install gradio && KIN_CKPT=kinyarwanda_small_v0.pt python app.py`.

## Training data
Assembled from free/open sources, cleaned to one sentence per line, language-filtered
(Kinyarwanda), and deduplicated. **~174M training tokens** (train split; + ~1.8M val,
~1.8M test).

| Source | License | Role |
|---|---|---|
| Kinyarwanda Wikipedia (rwwiki 20260601) | **CC BY-SA 4.0** + GFDL | ~3.5M tokens |
| MADLAD-400 `rw`, clean split (allenai) | **CC-BY-4.0** (underlying Common Crawl text keeps source copyrights) | ~156M tokens |
| Mbaza NLP Kinyarwanda monolingual v0.1.0 | **CC BY 4.0** | ~34M tokens |

Cleaning: Unicode NFC + whitespace normalization; a Kinyarwanda-stopword negative
language filter (KINNEWS stopword list — fastText `lid.176` cannot detect Kinyarwanda);
normalized-key deduplication (incl. cross-source overlap). Full provenance and per-step
counts are in the repo (`data/sources/SOURCES.md`, `PROGRESS.md`).

**Note:** KINNEWS was used **only for downstream evaluation**, not for pretraining.

## Training procedure
- From-scratch pretraining, ~40,000 steps, batch 12, context 512, ~2.75 epochs over
  the corpus. AdamW (no weight decay on biases/norms), linear warmup → cosine decay
  (peak LR 5e-4), gradient clipping 1.0, fp16 mixed precision.
- Hardware: a single free **Google Colab T4** GPU, across multiple ~5h sessions with
  checkpoint resume. Reproducible from scripts (`scripts/train.py`); fixed seed 123.

## Evaluation
Held-out test set (`test.bin`, 1.77M tokens):

- **Test perplexity: 58.14** (validation ~54.5 → generalizes, no overfitting).
- **Tokenizer fertility:** 1.47 tokens/word (vs GPT-2 3.07).

**Downstream — KINNEWS 14-class news classification** (fine-tuned base + head):

| Model | Test accuracy |
|---|---|
| **This model (fine-tuned)** | **87.38%** (macro-F1 79.4) |
| BiGRU (paper best, COLING 2020) | 88.65% |
| SVM | 88.53% |
| CNN | ~87.5% |
| random | ~7.1% |

Within ~1.3 points of the best specialized baseline. See `eval/RESULTS.md` for 20
generation samples awaiting native-speaker review.

## Limitations
- **Small & data-limited** (134M params, 174M tokens): perplexity plateaued near the
  end — this corpus/size is about at its ceiling. Expect repetition, occasional
  incoherence, and factual errors.
- Web/news-heavy data → skews toward news/formal register and current-affairs topics.
- Base model only — no instruction following, no safety tuning.
- Orthography edge cases and rare-word handling are imperfect.

## Bias, risks, and ethical considerations
- Trained on web and news text that can carry political, gender, regional, and other
  biases; outputs may reproduce them. It has **not** had native-speaker safety review
  for harmful content — this is a listed open task.
- May memorize/echo phrases from training data. Do not treat outputs as private,
  original, or factual.
- **Community:** this model is *for* Rwandans and should be evaluated and improved with
  Kinyarwanda speakers. Native-speaker fluency/factuality/harm review is pending.

## License (proposed — human to confirm before release)
The training corpus mixes **CC BY-SA 4.0** (Wikipedia, share-alike) with CC-BY-4.0
sources. Model-weight licensing derived from CC BY-SA training data is legally
unsettled; to be conservative and respectful of share-alike, the recommendation is to
release the weights under **CC BY-SA 4.0 with attribution** to all sources below.
**This is a decision for the project owner** (see repo `CLAUDE.md` §12) and ideally a
quick legal sanity check. Code in the repo is separate and can carry its own
(e.g. MIT/Apache) license.

## Citation & acknowledgements
- Architecture, training loop, tooling: **Sebastian Raschka**, *Build a Large Language
  Model (From Scratch)* — https://github.com/rasbt/LLMs-from-scratch
- Tokenization motivation: **KinyaBERT** (Nzeyimana & Niyongabo Rubungo, ACL 2022).
- Downstream benchmark + stopwords: **KINNEWS/KIRNEWS** (Niyongabo Rubungo et al.,
  COLING 2020).
- Data: **Wikimedia** (Kinyarwanda Wikipedia), **MADLAD-400** (Kudugunta et al., 2023),
  **Mbaza NLP** (Kinyarwanda monolingual), and the **Masakhane** community.

*Give Rwanda its model — responsibly, openly, and with Rwandans in the loop.*
