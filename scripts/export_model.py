"""
export_model.py — strip a training checkpoint down to a release-ready weights file.

Training checkpoints carry the optimizer state (~2x the model → ~1.6 GB). For sharing
you only need the weights + config. This drops the optimizer and (optionally) casts
to fp16, cutting ~1.6 GB down to ~269 MB (fp16) / ~538 MB (fp32).

Usage:
  python -m scripts.export_model --in checkpoints/kinyarwanda_small.pt \
      --out release/kinyarwanda_small_v0.pt --fp16
"""
import argparse
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="checkpoints/kinyarwanda_small.pt")
    ap.add_argument("--out", default="release/kinyarwanda_small_v0.pt")
    ap.add_argument("--fp16", action="store_true", help="cast weights to float16")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        sys.exit(f"Not found: {args.inp}")
    ck = torch.load(args.inp, map_location="cpu")
    sd = ck["model_state_dict"]
    if args.fp16:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model_state_dict": sd, "config": ck["config"],
                "step": ck.get("step"), "best_val": ck.get("best_val"),
                "dtype": "float16" if args.fp16 else "float32"}, args.out)
    src = os.path.getsize(args.inp) / 1e6
    dst = os.path.getsize(args.out) / 1e6
    print(f"{args.inp} ({src:.0f} MB) -> {args.out} ({dst:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
