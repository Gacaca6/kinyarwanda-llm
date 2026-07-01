"""
colab_train.py — run Kinyarwanda LLM pretraining on a free Google Colab GPU.

No phone verification required (unlike Kaggle). Paste this into a Colab cell AFTER
you've (1) set the runtime to a T4 GPU, (2) mounted Google Drive, and (3) uploaded
the token bins to Drive. See colab/README.md for the exact clicks.

Reads the bins from Drive and writes checkpoints to Drive, so training survives
Colab's ~12h session limit — just re-run this cell and --resume picks up the best
checkpoint. The training code is scripts/train.py (memmap data, warmup+cosine LR,
grad clip, mixed precision, val perplexity, resume).
"""
import os
import subprocess
import sys

REPO = "https://github.com/Gacaca6/kinyarwanda-llm.git"
# Drive locations (edit if you used different folders):
BINS = "/content/drive/MyDrive/kinyarwanda-llm/bins"
CKPT = "/content/drive/MyDrive/kinyarwanda-llm/checkpoints"

# --- get the code -----------------------------------------------------------
if not os.path.isdir("kinyarwanda-llm"):
    subprocess.run(["git", "clone", "--depth", "1", REPO], check=True)
os.chdir("kinyarwanda-llm")
# Colab already has torch (CUDA), numpy, tqdm — just need the tokenizer lib.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers"], check=True)

# --- sanity: GPU + Drive bins present ---------------------------------------
import torch  # noqa: E402
assert torch.cuda.is_available(), \
    "No GPU. Runtime -> Change runtime type -> Hardware accelerator: T4 GPU."
print("GPU:", torch.cuda.get_device_name(0))
for b in ("train.bin", "val.bin"):
    p = os.path.join(BINS, b)
    assert os.path.exists(p), \
        f"Missing {p}. Mount Drive and upload the bins there (see colab/README.md)."
    print(f"  {b}: {os.path.getsize(p)/1e6:.0f} MB")
os.makedirs(CKPT, exist_ok=True)

# --- train ------------------------------------------------------------------
# RWANDA_SMALL ~124M params, context 512. On a 16GB T4 start at batch 24; drop to
# 16/12 if you hit CUDA OOM. Data-limited (~174M tokens vs Chinchilla ~2.5B) -> train
# several epochs, watch val perplexity; best checkpoint is saved to Drive. ~14.5k
# steps ~= 1 epoch at batch 24 x 512. Re-run this cell after a disconnect to resume.
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
    "--out", CKPT,
    "--resume",
]
print("\n>", " ".join(cmd), "\n")
subprocess.run(cmd, check=True)

print(f"\nDone. Best checkpoint on Drive: {CKPT}/kinyarwanda_small.pt")
print("It persists across sessions — re-run this cell to continue training.")
