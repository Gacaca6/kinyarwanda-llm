"""
train.py — pretrain the Kinyarwanda GPT (Phase 4).

Upgraded from the book's chapter-5 loop for real pretraining:
  - reads pre-tokenized memmap data (data/processed/{train,val}.bin) -> low RAM,
    random-window batches (nanoGPT style)
  - AdamW with decoupled weight decay (no decay on biases/LayerNorm)
  - linear warmup -> cosine decay LR schedule, gradient clipping
  - mixed precision (torch.autocast + GradScaler) on CUDA
  - periodic validation loss -> perplexity, best-checkpoint saving, resume
  - tqdm progress, periodic Kinyarwanda sampling

Run from project root (build the .bin first with scripts/prepare_tokens.py):
  python -m scripts.prepare_tokens
  python -m scripts.train --model small --steps 20000        # GPU run
  python -m scripts.train --model tiny --steps 200 \
        --train-bin data/processed/val.bin --val-bin data/processed/test.bin
        # ^ tiny CPU smoke test (proves loss drops without the full corpus)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPTModel, generate                  # noqa: E402
from model.data_loader import KinyaTokenizer              # noqa: E402
from model.config import RWANDA_TINY, RWANDA_SMALL, TRAIN  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOK_PATH = "tokenizer/kinyarwanda_bpe/tokenizer.json"
CONFIGS = {"tiny": RWANDA_TINY, "small": RWANDA_SMALL}


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def loss_on(data, model, block, bs, device, iters):
    model.eval()
    losses = torch.zeros(iters)
    with torch.no_grad():
        for k in range(iters):
            x, y = get_batch(data, block, bs, device)
            logits = model(x)
            losses[k] = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), y.flatten())
    model.train()
    return losses.mean().item()


def lr_at(step, warmup, max_steps, lr, min_lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    if step > max_steps:
        return min_lr
    ratio = (step - warmup) / max(1, (max_steps - warmup))
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr - min_lr)


def make_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def sample(model, tok, device, block, prompt="U Rwanda", n=40):
    model.eval()
    ids = torch.tensor([tok.encode(prompt)], device=device)
    out = generate(model, ids, n, block, temperature=0.8, top_k=40)
    model.train()
    return tok.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["tiny", "small"], default="tiny")
    ap.add_argument("--train-bin", default="data/processed/train.bin")
    ap.add_argument("--val-bin", default="data/processed/val.bin")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None, help="context length")
    ap.add_argument("--lr", type=float, default=TRAIN["lr"])
    ap.add_argument("--min-lr", type=float, default=TRAIN["lr"] / 10)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--save-interval", type=int, default=2000,
                    help="save checkpoint every N steps (atomic). Bigger = fewer "
                         "slow Drive writes.")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = dict(CONFIGS[args.model])
    if args.block_size:
        cfg["context_length"] = args.block_size
    block = cfg["context_length"]
    bs = args.batch_size or TRAIN["batch_size"]
    warmup = args.warmup if args.warmup is not None else min(TRAIN["warmup_steps"],
                                                             max(1, args.steps // 20))

    for b in (args.train_bin, args.val_bin):
        if not os.path.exists(b):
            sys.exit(f"Missing {b}. Run: python -m scripts.prepare_tokens")
    train_data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")

    tok = KinyaTokenizer(TOK_PATH)
    model = GPTModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model, args.lr, TRAIN["weight_decay"])
    use_amp = device == "cuda"
    try:                                    # torch >= 2.4 (Colab)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):     # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_step, best_val = 0, float("inf")
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, f"kinyarwanda_{args.model}.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["opt_state_dict"])
        start_step = ck.get("step", 0)
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from step {start_step} (best_val={best_val:.3f})")

    print(f"device={device}  model={args.model}  params={n_params/1e6:.1f}M  "
          f"block={block}  batch={bs}  steps={args.steps}  warmup={warmup}  "
          f"train_tokens={len(train_data):,}")

    try:
        from tqdm import tqdm
        bar = tqdm(range(start_step, args.steps), initial=start_step,
                   total=args.steps, dynamic_ncols=True)
    except ImportError:
        bar = range(start_step, args.steps)

    model.train()
    t0 = time.time()
    for step in bar:
        lr = lr_at(step, warmup, args.steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, block, bs, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.grad_clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            vl = loss_on(val_data, model, block, bs, device, args.eval_iters)
            best_val = min(best_val, vl)
            ppl = math.exp(min(vl, 20))
            msg = f"step {step}  train_loss {loss.item():.3f}  val_loss {vl:.3f}  ppl {ppl:.1f}  lr {lr:.2e}"
            (bar.write if hasattr(bar, "write") else print)(msg)

        # Checkpoint on a coarser cadence (atomic write -> crash/disconnect safe,
        # and far fewer slow 1.6GB Drive writes than saving every eval).
        if step > start_step and (step % args.save_interval == 0 or step == args.steps - 1):
            tmp = ckpt_path + ".tmp"
            torch.save({"model_state_dict": model.state_dict(),
                        "opt_state_dict": opt.state_dict(),
                        "config": cfg, "step": step + 1, "best_val": best_val}, tmp)
            os.replace(tmp, ckpt_path)   # atomic: never leaves a half-written ckpt
            (bar.write if hasattr(bar, "write") else print)(
                f"  checkpoint saved @ step {step} -> {ckpt_path}")

    dt = time.time() - t0
    print(f"\ndone in {dt/60:.1f} min. best val_loss={best_val:.3f} "
          f"(ppl {math.exp(min(best_val,20)):.1f}). ckpt -> {ckpt_path}")
    print("sample:", sample(model, tok, device, block))


if __name__ == "__main__":
    sys.exit(main())
