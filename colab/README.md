# Training on a free Google Colab GPU (no phone verification)

Kaggle needs phone verification to unlock its GPU; Colab does **not** (no phone, no
card). Colab + Google Drive also gives **persistent checkpoints**, which matters
because this data-limited run needs several epochs across multiple sessions.

## The easy way — one cell, zero upload (recommended)

You do **not** need to upload the 348 MB corpus. [`colab_bootstrap.py`](colab_bootstrap.py)
rebuilds it from the public sources *inside Colab* (fast download there) the first
time, caches the bins to your Drive, then trains. Later runs reuse the cached bins.

1. Open **https://colab.research.google.com** → **New notebook**.
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
3. **Cell 1 — mount Drive** (run it, approve the popup):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Cell 2 — everything:** paste the contents of
   [`colab_bootstrap.py`](colab_bootstrap.py) and run. First run: builds + caches the
   corpus (~15–25 min) then trains. Later runs: straight to training. Best checkpoint
   is saved to `MyDrive/kinyarwanda-llm/checkpoints/kinyarwanda_small.pt`.

That's it — no manual data handling.

## The other way — pre-upload the bins yourself

If you'd rather build the bins locally and upload them (skips the in-Colab rebuild):
```bash
python -m scripts.prepare_tokens     # writes data/processed/{train,val,test}.bin
```
Upload them to `MyDrive/kinyarwanda-llm/bins/` on drive.google.com, then use
[`colab_train.py`](colab_train.py) as Cell 2 instead of the bootstrap.

## Notes / knobs

- **Resume across sessions.** Checkpoints go to Drive, so if Colab disconnects
  (~12h limit, or idle timeout), just re-run Cell 2 — `--resume` continues from the
  best checkpoint. Keep the browser tab active to avoid idle disconnects.
- **CUDA OOM?** Lower `--batch-size` in `colab_train.py` (24 → 16 → 12).
- **Sanity first.** For a quick throughput check before the long run, temporarily set
  `--model tiny --steps 2000`.
- **Data-limited.** ~174M train tokens vs Chinchilla-optimal ~2.5B for a 124M model,
  so train multiple epochs and stop when val perplexity stops improving (the best
  checkpoint is saved automatically). ~14.5k steps ≈ 1 epoch at batch 24 × 512.
- When you have a checkpoint, we move to **Phase 5**: perplexity on `test.bin`,
  downstream eval (KINNEWS/MasakhaNER), and native-speaker review.

## Other phone-free options

If Colab's limits chafe: **SageMaker Studio Lab** (free T4, 4h sessions, no card),
**Lightning AI**, or **Paperspace Gradient** — all avoid Kaggle's phone gate. The
same `train.py` runs on any of them; only the bin path and checkpoint path change.
