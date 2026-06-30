# Training on a free Kaggle GPU

This project has no local GPU, so pretraining (Phase 4) runs on Kaggle's free
GPU (~30 hrs/week, T4 16GB or P100). The CPU smoke test already proved the loop
works (loss 10.5 → 7.6 in 150 steps); Kaggle is for the real RWANDA_SMALL run.

## One-time: upload the token bins as a Kaggle Dataset

The cleaned corpus and its tokenized `.bin` files are **not** in git (too large).
Build them locally, then upload to Kaggle once (they persist across notebooks).

```bash
# locally, from the repo root:
python -m scripts.prepare_tokens          # writes data/processed/{train,val,test}.bin
```

1. Go to kaggle.com → **Datasets → New Dataset**.
2. Upload `data/processed/train.bin`, `val.bin`, `test.bin`.
3. Name it (e.g. `kinyarwanda-bins`). Note the slug — the input path becomes
   `/kaggle/input/kinyarwanda-bins/`.

(Alternative: upload `data/processed/train.txt` instead and have the notebook run
`python -m scripts.prepare_tokens` itself — bigger upload, but rebuilds from text.)

## Each training session

1. **New Notebook** on Kaggle.
2. **Settings → Accelerator → GPU** (T4 or P100). **Settings → Internet → On**
   (needed to clone the repo + pip install).
3. **Add Input → Datasets →** your `kinyarwanda-bins`.
4. Paste the contents of [`kaggle_train.py`](kaggle_train.py) into a cell (edit the
   `BINS = ...` path to match your dataset slug) and run. It will:
   - clone this repo (model code, 32k tokenizer, `scripts/train.py`),
   - verify the GPU and bins,
   - train RWANDA_SMALL, logging val perplexity and saving the best checkpoint to
     `/kaggle/working/checkpoints/kinyarwanda_small.pt`.

## Notes / knobs

- **OOM?** Lower `--batch-size` (24 → 16 → 12). RWANDA_SMALL is ~124M params at
  context 512.
- **Data-limited.** ~178M training tokens vs Chinchilla-optimal ~2.5B for 124M
  params, so expect to train several epochs. The script saves the **best** (lowest
  val-loss) checkpoint — stop when val perplexity stops improving (overfitting).
- **Sessions end (~9–12h).** To continue, download the checkpoint, upload it as a
  dataset, attach it, copy it into `/kaggle/working/checkpoints/`, and the script's
  `--resume` will pick up where it left off.
- **Start smaller first.** A quick sanity run on Kaggle:
  `--model tiny --steps 2000` to confirm GPU throughput before the long run.
- Fixed seeds + logged config make runs reproducible; record results in
  `eval/RESULTS.md` (Phase 5).
