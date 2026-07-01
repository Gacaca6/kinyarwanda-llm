"""
evaluate.py — Phase 5 intrinsic evaluation of a trained Kinyarwanda GPT.

Computes:
  1. Held-out perplexity on data/processed/test.bin (proper corpus perplexity over
     non-overlapping windows, token-weighted).
  2. Tokenizer fertility (tokens/word) on the test text.
  3. N generation samples from Kinyarwanda seed prompts, for native-speaker review.

Writes a section to eval/RESULTS.md. Runs on CPU (no GPU quota needed) or GPU.

Usage (from repo root):
  python -m eval.evaluate --ckpt checkpoints/kinyarwanda_small.pt
  # on Colab with the checkpoint on Drive:
  python -m eval.evaluate --ckpt /content/drive/MyDrive/kinyarwanda-llm/checkpoints/kinyarwanda_small.pt
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPTModel, generate            # noqa: E402
from model.data_loader import KinyaTokenizer        # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOK_PATH = "tokenizer/kinyarwanda_bpe/tokenizer.json"
TEST_BIN = "data/processed/test.bin"
TEST_TXT = "data/processed/test.txt"
RESULTS = "eval/RESULTS.md"

PROMPTS = [
    "U Rwanda", "Abanyarwanda", "Mu mujyi wa Kigali", "Amateka y'u Rwanda",
    "Uburezi ni", "Ubuvuzi mu Rwanda", "Ikoranabuhanga", "Umuganda ni",
    "Ubukungu bw'u Rwanda", "Urubyiruko rw'u Rwanda", "Imyaka ishize",
    "Perezida wa Repubulika", "Mu karere k'Afurika", "Ubuhinzi n'ubworozi",
    "Umuco nyarwanda", "Guverinoma y'u Rwanda", "Abagore n'abakobwa",
    "Mu rwego rwo guteza imbere", "Ikinyarwanda ni ururimi", "Isi yose",
]


@torch.no_grad()
def perplexity(data, model, block, device, batch_size, max_blocks=0):
    """Token-weighted perplexity over non-overlapping windows of the test set."""
    n_blocks = (len(data) - 1) // block
    if max_blocks:
        n_blocks = min(n_blocks, max_blocks)
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tok = 0.0, 0
    xs, ys = [], []
    for b in range(n_blocks):
        i = b * block
        xs.append(torch.from_numpy(data[i:i + block].astype(np.int64)))
        ys.append(torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)))
        if len(xs) == batch_size or b == n_blocks - 1:
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
            logits = model(x)
            total_loss += ce(logits.flatten(0, 1), y.flatten()).item()
            total_tok += y.numel()
            xs, ys = [], []
    mean = total_loss / total_tok
    return math.exp(mean), mean, total_tok


def fertility(txt_path, tok, limit_lines=100000):
    if not os.path.exists(txt_path):
        return None
    n_tok = n_word = 0
    for i, line in enumerate(open(txt_path, encoding="utf-8")):
        if i >= limit_lines:
            break
        line = line.strip()
        if line:
            n_tok += len(tok.encode(line))
            n_word += len(line.split())
    return n_tok / n_word if n_word else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/kinyarwanda_small.pt")
    ap.add_argument("--test-bin", default=TEST_BIN)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="cap test windows (0=all; use e.g. 400 for a quick CPU run)")
    ap.add_argument("--num-samples", type=int, default=len(PROMPTS))
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.ckpt):
        sys.exit(f"Checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    block = cfg["context_length"]
    model = GPTModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tok = KinyaTokenizer(TOK_PATH)
    n_params = sum(p.numel() for p in model.parameters())
    trained_step = ckpt.get("step", "?")
    print(f"device={device}  params={n_params/1e6:.1f}M  trained_step={trained_step}")

    # 1) perplexity
    data = np.memmap(args.test_bin, dtype=np.uint16, mode="r")
    ppl, mean_loss, n_tok = perplexity(data, model, block, device,
                                       args.batch_size, args.max_blocks)
    print(f"test perplexity = {ppl:.2f}  (loss {mean_loss:.3f}, {n_tok:,} tokens)")

    # 2) fertility
    fert = fertility(TEST_TXT, tok)
    if fert:
        print(f"tokenizer fertility on test = {fert:.2f} tokens/word")

    # 3) generation samples
    print("\n--- generation samples ---")
    samples = []
    for p in PROMPTS[:args.num_samples]:
        ids = torch.tensor([tok.encode(p)], device=device)
        out = generate(model, ids, args.max_new_tokens, block,
                       temperature=args.temperature, top_k=args.top_k,
                       eos_id=tok.eot_id)
        text = tok.decode(out[0].tolist())
        samples.append((p, text))
        print(f"[{p}] {text}\n")

    # write RESULTS.md
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as r:
        r.write("# Evaluation results (Phase 5)\n\n")
        r.write(f"Model: RWANDA_SMALL, {n_params/1e6:.1f}M params, trained to step "
                f"{trained_step}.\n\n")
        r.write("## Intrinsic\n\n")
        r.write(f"- **Held-out test perplexity:** {ppl:.2f} "
                f"(over {n_tok:,} tokens of `test.bin`)\n")
        if fert:
            r.write(f"- **Tokenizer fertility (test):** {fert:.2f} tokens/word "
                    f"(vs GPT-2 ~3.07)\n")
        r.write("\n## Generation samples (for native-speaker review)\n\n")
        r.write("Seed prompt in **bold**; the rest is model continuation "
                f"(temp {args.temperature}, top-k {args.top_k}).\n\n")
        for p, text in samples:
            cont = text[len(p):].strip() if text.startswith(p) else text
            r.write(f"- **{p}** {cont}\n")
        r.write("\n> Native Kinyarwanda speakers: please rate each for fluency "
                "(grammar) and factuality, and flag anything harmful or biased.\n")
    print(f"\nWrote {RESULTS}")


if __name__ == "__main__":
    sys.exit(main())
