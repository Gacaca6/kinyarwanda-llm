"""
train.py — pretrain the Kinyarwanda GPT.

Training loop adapted from "Build a Large Language Model (From Scratch)"
(chapter 5: gpt_train.py). Run from the project root:

    python -m scripts.train            # uses RWANDA_TINY + the prepared corpus

It splits the corpus 90/10 into train/val, trains, prints train/val loss,
periodically samples Kinyarwanda text, and saves the model + a loss curve.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPTModel, generate                       # noqa: E402
from model.data_loader import KinyaTokenizer, create_dataloader  # noqa: E402
from model.config import RWANDA_TINY, TRAIN                     # noqa: E402

CORPUS = "data/kinyarwanda_corpus.txt"
TOK_PATH = "tokenizer/kinyarwanda_bpe/tokenizer.json"
OUT = "checkpoints"


def calc_loss_batch(inp, tgt, model, device):
    inp, tgt = inp.to(device), tgt.to(device)
    logits = model(inp)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), tgt.flatten())


def calc_loss_loader(loader, model, device, num_batches=None):
    total, n = 0.0, 0
    if len(loader) == 0:
        return float("nan")
    num_batches = len(loader) if num_batches is None else min(num_batches, len(loader))
    for i, (inp, tgt) in enumerate(loader):
        if i >= num_batches:
            break
        with torch.no_grad():
            total += calc_loss_batch(inp, tgt, model, device).item()
        n += 1
    return total / n


def sample(model, tokenizer, device, prompt="U Rwanda", max_new_tokens=30):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = generate(model, ids, max_new_tokens,
                   RWANDA_TINY["context_length"], temperature=0.8, top_k=40)
    model.train()
    return tokenizer.decode(out[0].tolist())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)

    tokenizer = KinyaTokenizer(TOK_PATH)
    text = open(CORPUS, encoding="utf-8").read()
    split = int(0.9 * len(text))
    train_txt, val_txt = text[:split], text[split:]

    cl, st, bs = (RWANDA_TINY["context_length"], TRAIN["stride"], TRAIN["batch_size"])
    train_loader = create_dataloader(train_txt, tokenizer, bs, cl, st, shuffle=True)
    val_loader = create_dataloader(val_txt, tokenizer, bs, cl, st, shuffle=False)

    model = GPTModel(RWANDA_TINY).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params/1e6:.1f}M  "
          f"train_batches={len(train_loader)}  val_batches={len(val_loader)}")

    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"],
                            weight_decay=TRAIN["weight_decay"])

    train_losses, val_losses, step = [], [], 0
    for epoch in range(TRAIN["num_epochs"]):
        model.train()
        for inp, tgt in train_loader:
            opt.zero_grad()
            loss = calc_loss_batch(inp, tgt, model, device)
            loss.backward()
            opt.step()
            step += 1
            if step % TRAIN["eval_freq"] == 0:
                tr = calc_loss_loader(train_loader, model, device, TRAIN["eval_iter"])
                vl = calc_loss_loader(val_loader, model, device, TRAIN["eval_iter"])
                train_losses.append(tr)
                val_losses.append(vl)
                print(f"ep{epoch+1} step{step:>4}  train={tr:.3f}  val={vl:.3f}")
        print("  sample:", sample(model, tokenizer, device))

    os.makedirs(OUT, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": RWANDA_TINY},
               os.path.join(OUT, "kinyarwanda_gpt.pt"))
    print(f"Saved -> {OUT}/kinyarwanda_gpt.pt")


if __name__ == "__main__":
    main()
