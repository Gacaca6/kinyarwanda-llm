# Tokenizer sweep (Phase 3)

**Decision: vocab = 32,000** (promoted to `tokenizer/kinyarwanda_bpe/`,
`VOCAB_SIZE` updated in `model/config.py`).

Rationale: fertility gains diminish sharply (16k→32k saves 7.5% tokens/word; 32k→50k
only 3.4%) while the embedding + output head cost grows linearly (vocab × emb_dim × 2:
at emb_dim 768 that's 24.6M / 49.2M / 76.8M params for 16k/32k/50k). 50k would make
embeddings ~half of a RWANDA_SMALL model for little gain. At ~178M training tokens each
of 32k types is seen ~5,500×, plenty to learn good embeddings. 32k is the
KinyaBERT-class sweet spot.

Trained on `data/processed/train.txt`, evaluated on `data/processed/val.txt` (1,165,183 words).

| Tokenizer | vocab | tokens/word (fertility) | chars/token | vs GPT-2 |
|---|---|---|---|---|
| GPT-2 (English BPE) | 50257 | 3.07 | 2.33 | 1.00x |
| Kinyarwanda BPE | 16000 | 1.59 | 4.49 | 1.93x |
| Kinyarwanda BPE | 32000 | 1.47 | 4.85 | 2.08x |
| Kinyarwanda BPE | 50000 | 1.42 | 5.03 | 2.16x |

Lower fertility is better. Bigger vocab lowers fertility but enlarges the model's embedding + output head (vocab x emb_dim x2).

## Example: agglutinative words, token counts by vocab

```
ntibazabikora    GPT-2=6  v16000=3  v32000=3  v50000=3
turabashimira    GPT-2=5  v16000=3  v32000=3  v50000=3
abanyarwanda     GPT-2=5  v16000=1  v32000=1  v50000=1
uburezi          GPT-2=3  v16000=1  v32000=1  v50000=1
umuganda         GPT-2=3  v16000=1  v32000=1  v50000=1
icyorezo         GPT-2=3  v16000=1  v32000=1  v50000=1
umunyarwanda     GPT-2=6  v16000=1  v32000=1  v50000=1
ubwiyunge        GPT-2=5  v16000=1  v32000=1  v50000=1
guhanga          GPT-2=3  v16000=2  v32000=2  v50000=2
```
