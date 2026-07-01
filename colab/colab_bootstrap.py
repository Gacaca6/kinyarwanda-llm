"""
colab_bootstrap.py — ONE-CELL, zero-upload Kinyarwanda LLM training on a free Colab GPU.

What it does, in order:
  1. clone the repo + install deps
  2. bins on Drive already?  -> copy them local and train
     not yet?               -> rebuild the whole corpus from the public sources
                               (Wikipedia + MADLAD-400 rw + mbaza), tokenize, and
                               CACHE the bins to Drive so next time is instant
  3. train RWANDA_SMALL, saving the best checkpoint to Drive (survives disconnects;
     re-run this cell and --resume continues).

You do NOT need to upload the 348 MB corpus yourself — Colab downloads the sources
directly (fast) the first time only.

BEFORE running this cell:
  - Runtime -> Change runtime type -> T4 GPU
  - Run a first cell to mount Drive:
        from google.colab import drive; drive.mount('/content/drive')
Then paste + run this file.
"""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/Gacaca6/kinyarwanda-llm.git"
DRIVE = "/content/drive/MyDrive/kinyarwanda-llm"
BINS_DRIVE = f"{DRIVE}/bins"
CKPT = f"{DRIVE}/checkpoints"
SPLITS = ("train.bin", "val.bin", "test.bin")


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# reduce CUDA fragmentation OOMs on the 16GB T4
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# 1) code + deps ------------------------------------------------------------
if os.path.isdir("kinyarwanda-llm"):
    subprocess.run(["git", "-C", "kinyarwanda-llm", "pull", "--ff-only"])  # get fixes
else:
    run(["git", "clone", "--depth", "1", REPO])
os.chdir("kinyarwanda-llm")
run([sys.executable, "-m", "pip", "install", "-q",
     "tokenizers", "mwparserfromhell", "pyarrow", "numpy", "tqdm"])

import torch  # noqa: E402
assert torch.cuda.is_available(), \
    "No GPU. Runtime -> Change runtime type -> Hardware accelerator: T4 GPU."
assert os.path.isdir("/content/drive/MyDrive"), \
    "Drive not mounted. Run in a prior cell: from google.colab import drive; drive.mount('/content/drive')"
print("GPU:", torch.cuda.get_device_name(0), flush=True)

os.makedirs(BINS_DRIVE, exist_ok=True)
os.makedirs(CKPT, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# 2) bins: reuse from Drive, or build once and cache ------------------------
if all(os.path.exists(f"{BINS_DRIVE}/{b}") for b in SPLITS):
    print("Found cached bins on Drive — copying local (fast random reads).", flush=True)
    for b in SPLITS:
        shutil.copy(f"{BINS_DRIVE}/{b}", f"data/processed/{b}")
else:
    print("First run: building the corpus from public sources (~15-25 min)...", flush=True)
    run([sys.executable, "data/prepare_wikipedia.py"])          # ~12 MB
    run([sys.executable, "data/prepare_madlad.py", "--split", "clean"])  # 277 MB
    run([sys.executable, "data/prepare_mbaza.py"])              # 191 MB
    run([sys.executable, "data/clean_corpus.py"])               # filter+dedup+split
    run([sys.executable, "-m", "scripts.prepare_tokens"])       # -> *.bin
    print("Caching bins to Drive for next time...", flush=True)
    for b in SPLITS:
        shutil.copy(f"data/processed/{b}", f"{BINS_DRIVE}/{b}")

for b in ("train.bin", "val.bin"):
    print(f"  {b}: {os.path.getsize(f'data/processed/{b}')/1e6:.0f} MB", flush=True)

# 3) train ------------------------------------------------------------------
# RWANDA_SMALL ~124M params, ctx 512. On a 16GB T4 start at batch 24; drop to 16/12
# if you hit CUDA OOM. ~174M train tokens (data-limited) -> train several epochs and
# watch val perplexity; best checkpoint auto-saved to Drive. Re-run this cell after a
# disconnect to resume. ~14.5k steps ~= 1 epoch at batch 24 x 512.
run([sys.executable, "-m", "scripts.train",
     "--model", "small",
     "--train-bin", "data/processed/train.bin",
     "--val-bin", "data/processed/val.bin",
     "--steps", "40000",
     "--batch-size", "12",
     "--lr", "5e-4",
     "--warmup", "1000",
     "--eval-interval", "500",
     "--eval-iters", "100",
     "--out", CKPT,
     "--resume"])

print(f"\nDone. Best checkpoint on Drive: {CKPT}/kinyarwanda_small.pt", flush=True)
print("Bins cached at", BINS_DRIVE, "- re-run this cell anytime to continue training.",
      flush=True)
