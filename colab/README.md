# Training on a free Google Colab GPU (no phone verification)

Kaggle needs phone verification to unlock its GPU; Colab does **not** (no phone, no
card). The pipeline is identical — same `train.py`, same `.bin` files. Colab + Google
Drive also gives you **persistent checkpoints**, which matters because this
data-limited run needs several epochs across multiple sessions.

## One-time: build the bins and put them on Drive

```bash
# locally, from the repo root:
python -m scripts.prepare_tokens     # writes data/processed/{train,val,test}.bin
```

Then upload to Google Drive (drive.google.com), creating this folder:

```
MyDrive/kinyarwanda-llm/bins/train.bin   (~348 MB)
MyDrive/kinyarwanda-llm/bins/val.bin
MyDrive/kinyarwanda-llm/bins/test.bin
```

(train.bin is 348 MB; Drive's free 15 GB is plenty. Upload can take a while.)

## Each training session

1. Open **https://colab.research.google.com** → **New notebook**.
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
3. **Cell 1 — mount Drive** (run it, approve the popup):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. **Cell 2 — train:** paste the contents of [`colab_train.py`](colab_train.py) and run.
   It clones the repo, installs `tokenizers`, checks the GPU + bins, and trains
   RWANDA_SMALL, saving the best checkpoint to
   `MyDrive/kinyarwanda-llm/checkpoints/kinyarwanda_small.pt`.

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
