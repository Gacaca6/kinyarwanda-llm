# Evaluation results (Phase 5)

**Model:** RWANDA_SMALL — 134.6M params, GPT (12 layers, 768 dim, ctx 512), trained
from scratch on ~174M tokens of clean Kinyarwanda (Wikipedia + MADLAD-400 + mbaza),
32k Kinyarwanda BPE tokenizer. Base (text-completion) model, v0.

## Intrinsic

| Metric | Value | Note |
|---|---|---|
| **Held-out test perplexity** | **58.14** | over the full 1,773,056-token `test.bin`; val was ~54.5 → generalizes, no overfitting |
| Tokenizer fertility (test) | 1.47 tokens/word | vs GPT-2's 3.07 → 2.08× more efficient (Phase 3) |

## Downstream — KINNEWS news classification (14 classes)

Fine-tuned the base model with a classification head (last-real-token → 14 classes),
3 epochs, on KINNEWS (Niyongabo et al., COLING 2020): 17,014 train / 4,254 test.

| Model | Test accuracy | macro-F1 |
|---|---|---|
| **Ours (134M base, fine-tuned)** | **87.38%** | 79.39 |
| BiGRU (W2V-Kin-50) — paper best | 88.65% | — |
| SVM (TF-IDF) | 88.53% | — |
| CNN (W2V-Kin-50) | ~87.55% | — |
| char-CNN | 71.70% | — |
| random chance | ~7.1% | — |

Our general-purpose base model lands **within ~1.3 points of the best specialized
baseline and matches the CNN baseline** — evidence the pretraining learned useful
Kinyarwanda representations, not just surface statistics. (Per-epoch: 86.69 → 87.00 →
87.38%.) macro-F1 < accuracy because several of the 14 classes are infrequent.

## Generation samples (for native-speaker review)

Seed prompt in **bold**; the rest is the model's continuation (temperature 0.8,
top-k 40). Native Kinyarwanda speakers: please rate each for **fluency** (grammar) and
**factuality**, and flag anything harmful or biased.

- **U Rwanda** ni rwo rwa mbere ruhuriye ku mukino wa mbere ruhuza amakipe y'ibihugu 6 iwayo muri Afurika.
- **Abanyarwanda** baba muri Leta Zunze Ubumwe za Amerika, u Buholandi, u Burayi na Denmark byashyizeho ibihano bikarishye.
- **Mu mujyi wa Kigali** rwagati, mu nzu ya mbere habereye impanuka y'imodoka nshya y'abapolisi bari barimo mu kazi, abagera kuri 20 bari bayirimo.
- **Amateka y'u Rwanda** mu bihe byashize, ariko n'ubwo atari bwo bwa mbere yari afite inkomoko yo ku rwego rwo hejuru, yaje guhinduka izina ry'ubusizi, ubuvanganzo, ubuvanganzo, ubuvanganzo, umuco n'izindi. *(note: repetition artifact)*
- **Uburezi ni** inshingano zacu gukomeza kuba izingiro ry'iterambere ry'Igihugu cyacu."
- **Ubuvuzi mu Rwanda** bari mu byiciro 3, birimo icya kabiri cy'impunzi zavuye mu bihugu by'Afurika y'Iburasirazuba, muri byo harimo n'icya gatatu cy'impunzi z'Abarundi.
- **Ikoranabuhanga** (ITU) rifite umwanya wihariye (Foto Kayitare J.P)
- **Umuganda ni** igikorwa cyiza kuko kiri mu nshingano z'abaturage n'abayobozi b'ibigo by'amashuri kandi n'ibigo by'amashuri byose bya Leta, kugira ngo abana bafite ibyo bibazo babashe kujya ku mashuri".
- **Ubukungu bw'u Rwanda** (RSE)
- **Urubyiruko rw'u Rwanda**
- **Imyaka ishize** habaye ubwiyongere bw'abaturage b'igihugu mu bijyanye n'ibiribwa no gukoresha ibikoresho by'ibanze.
- **Perezida wa Repubulika** Paul Kagame, yavuze ko kuba umuturage w'u Rwanda ari ahantu heza, bisaba imbaraga nyinshi kandi abantu bagatekereza neza aho bakorera, ari na yo mpamvu hakenewe imbaraga nyinshi mu guteza imbere igihugu.
- **Mu karere k'Afurika** y'Iburasirazuba hari haraheze amezi atatu gusa muri Afurika y'iburasirazuba.
- **Ubuhinzi n'ubworozi**, buhinzi n'ubworozi bw'inka.
- **Umuco nyarwanda** - IGIHE.com
- **Guverinoma y'u Rwanda** yatangiye gahunda yo kuvugurura ubuhinzi, gutera amashyamba no gufata neza amashyamba mu buryo bworoshye kandi bukozwe neza, ndetse no kuhira imyaka, aho u Rwanda rugeze rurushaho gutera imbere, ibi byose bikaba biri mu rwego rwo guteza imbere ubuhinzi bw'ibihaza, ubworozi
- **Abagore n'abakobwa** bagera kuri 55 mu Karere ka Huye bamaze guhugurwa mu bijyanye no kwihangira imirimo.
- **Mu rwego rwo guteza imbere** ihame ry'uburinganire muri gahunda za Leta harimo gukangurira abagore n'urubyiruko guharanira ihame ry'uburinganire n'ubwuzuzanye mu miryango, kubaka ubushobozi bw'umugore mu gihugu no hanze yacyo.
- **Ikinyarwanda ni ururimi** ruvugwa mu mashuri n'amashuri makuru na kaminuza.
- **Isi yose** izibanda ku bikorwa by'ubukerarugendo bwo mu Rwanda, u Rwanda rukaba rwarabonye imirimo myinshi yo gucukura amabuye y'agaciro mu rwego rwo guca burundu ikibazo cy'ibura ry'amaso mu bihe by'imvura. *(note: last clause is incoherent — factuality flag)*

### Human evaluation (to be completed by native speakers)
- [ ] Fluency rating (1–5) per sample
- [ ] Factuality / hallucination flags
- [ ] Harmful / biased content flags
- [ ] Overall v0 assessment

## Reproduce

```bash
python -m eval.evaluate        --ckpt <checkpoint>   # intrinsic + samples
python -m eval.finetune_kinnews --ckpt <checkpoint>   # downstream, --append-results
```
