"""
kaggle_train.py — run Kinyarwanda LLM pretraining on a free Kaggle GPU.

Paste this into a Kaggle Notebook (GPU T4/P100 + Internet ON) after attaching the
token-bins dataset (see kaggle/README.md). It clones the repo for the model code +
tokenizer, points training at the uploaded .bin files, and trains RWANDA_SMALL while
logging validation perplexity and saving the best checkpoint to /kaggle/working.

The training code itself is scripts/train.py (memmap data, warmup+cosine LR, grad
clip, mixed precision, checkpoint/resume). This file is just Kaggle orchestration.
"""
import os
import subprocess
import sys

REPO = "https://github.com/Gacaca6/kinyarwanda-llm.git"
# Edit to match YOUR attached dataset folder under /kaggle/input/<slug>/ :
BINS = "/kaggle/input/kinyarwanda-bins"

# --- get the code -----------------------------------------------------------
if not os.path.isdir("kinyarwanda-llm"):
    subprocess.run(["git", "clone", "--depth", "1", REPO], check=True)
os.chdir("kinyarwanda-llm")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "tokenizers", "numpy", "tqdm"], check=True)

# --- sanity: GPU + bins present ---------------------------------------------
import torch  # noqa: E402
assert torch.cuda.is_available(), "Enable GPU in Kaggle (Settings -> Accelerator)."
print("GPU:", torch.cuda.get_device_name(0))
for b in ("train.bin", "val.bin"):
    p = os.path.join(BINS, b)
    assert os.path.exists(p), f"Missing {p} — attach the bins dataset (see README)."
    print(f"  {b}: {os.path.getsize(p)/1e6:.0f} MB")

# --- train ------------------------------------------------------------------
# RWANDA_SMALL ~124M params, context 512. On a 16GB T4 start at batch 24; if you
# hit CUDA OOM, drop --batch-size (16, then 12). We are data-limited (~178M tokens
# vs Chinchilla-optimal ~2.5B), so train several epochs and WATCH val perplexity —
# the best checkpoint (lowest val loss) is saved automatically; stop if val stops
# improving (overfitting). ~14.5k steps ≈ 1 epoch at batch 24 x 512.
cmd = [
    sys.executable, "-m", "scripts.train",
    "--model", "small",
    "--train-bin", f"{BINS}/train.bin",
    "--val-bin", f"{BINS}/val.bin",
    "--steps", "40000",
    "--batch-size", "24",
    "--lr", "5e-4",
    "--warmup", "1000",
    "--eval-interval", "500",
    "--eval-iters", "100",
    "--out", "/kaggle/working/checkpoints",
    "--resume",                      # resumes if a checkpoint is present
]
print("\n>", " ".join(cmd), "\n")
subprocess.run(cmd, check=True)

print("\nDone. Best checkpoint at /kaggle/working/checkpoints/kinyarwanda_small.pt")
print("Download it from the notebook's Output tab, or save the notebook to persist it.")
