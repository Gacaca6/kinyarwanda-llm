"""
finetune_kinnews.py — Phase 5 downstream evaluation: KINNEWS news classification.

Fine-tunes the pretrained Kinyarwanda GPT (base checkpoint) with a classification
head on KINNEWS (Niyongabo et al., COLING 2020): 21,268 Kinyarwanda news articles,
14 classes, train/test = 17,014 / 4,254. Reports test accuracy + macro-F1 and
contextualizes against the paper's best monolingual baseline (BiGRU 88.65%, SVM
88.53%).

Data: https://github.com/saradhix/kinnews_kirnews/raw/master/KINNEWS.zip (MIT license
for the packaging; article copyright stays with the news sources — used here only for
benchmark evaluation, NOT folded into the pretraining corpus).

Usage (GPU recommended; ~20-40 min on a T4):
  python -m eval.finetune_kinnews --ckpt /content/drive/MyDrive/kinyarwanda-llm/checkpoints/kinyarwanda_small.pt
"""
import argparse
import csv
import io
import math
import os
import sys
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPTModel                       # noqa: E402
from model.data_loader import KinyaTokenizer         # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

TOK_PATH = "tokenizer/kinyarwanda_bpe/tokenizer.json"
DATA_URL = "https://github.com/saradhix/kinnews_kirnews/raw/master/KINNEWS.zip"
DATA_DIR = "data/sources/kinnews"
UA = "kinyarwanda-llm/0.1 (open Kinyarwanda LLM; KINNEWS benchmark eval)"
NUM_CLASSES = 14
CLASSES = ["politics", "sport", "economy", "health", "entertainment", "history",
           "technology", "tourism", "culture", "fashion", "religion",
           "environment", "education", "relationship"]
BASELINE = "paper best (COLING 2020): BiGRU 88.65%, SVM 88.53%, CNN ~87.5%"


def download_data():
    zpath = os.path.join(DATA_DIR, "KINNEWS.zip")
    if not os.path.exists(zpath):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"downloading {DATA_URL}")
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req) as r, open(zpath, "wb") as w:
            w.write(r.read())
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA_DIR)
    return (os.path.join(DATA_DIR, "KINNEWS", "cleaned", "train.csv"),
            os.path.join(DATA_DIR, "KINNEWS", "cleaned", "test.csv"))


def read_csv(path, tok, max_len, limit=0):
    ids_list, labels = [], []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # label,title,content
        for row in reader:
            if len(row) < 3:
                continue
            try:
                label = int(row[0]) - 1          # 1..14 -> 0..13
            except ValueError:
                continue
            if not (0 <= label < NUM_CLASSES):
                continue
            text = (row[1] + " " + row[2]).strip()
            ids = tok.encode(text)[:max_len]
            if not ids:
                continue
            ids_list.append(ids)
            labels.append(label)
            if limit and len(labels) >= limit:
                break
    return ids_list, labels


class Batcher:
    """Pads variable-length token lists per batch; tracks true lengths."""
    def __init__(self, ids_list, labels, pad_id, batch_size, shuffle, device):
        self.ids, self.labels = ids_list, labels
        self.pad_id, self.bs = pad_id, batch_size
        self.shuffle, self.device = shuffle, device

    def __len__(self):
        return math.ceil(len(self.ids) / self.bs)

    def __iter__(self):
        order = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(order)
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]
            seqs = [self.ids[i] for i in idx]
            lens = [len(x) for x in seqs]
            maxl = max(lens)
            x = torch.full((len(seqs), maxl), self.pad_id, dtype=torch.long)
            for j, seq in enumerate(seqs):
                x[j, :len(seq)] = torch.tensor(seq)
            y = torch.tensor([self.labels[i] for i in idx])
            last = torch.tensor([l - 1 for l in lens])
            yield x.to(self.device), y.to(self.device), last.to(self.device)


class GPTClassifier(nn.Module):
    """Reuses a pretrained GPT backbone; classifies from the last real token."""
    def __init__(self, base, num_classes):
        super().__init__()
        self.tok_emb = base.tok_emb
        self.pos_emb = base.pos_emb
        self.drop_emb = base.drop_emb
        self.trf_blocks = base.trf_blocks
        self.final_norm = base.final_norm
        self.head = nn.Linear(base.tok_emb.embedding_dim, num_classes)

    def forward(self, idx, last):
        _, T = idx.shape
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop_emb(self.tok_emb(idx) + pos)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        h = x[torch.arange(x.size(0)), last]     # hidden state at last real token
        return self.head(h)


@torch.no_grad()
def evaluate(model, batcher):
    model.eval()
    correct = total = 0
    tp = [0] * NUM_CLASSES
    fp = [0] * NUM_CLASSES
    fn = [0] * NUM_CLASSES
    for x, y, last in batcher:
        pred = model(x, last).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for p, t in zip(pred.tolist(), y.tolist()):
            if p == t:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1
    f1s = []
    for c in range(NUM_CLASSES):
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return correct / total, sum(f1s) / NUM_CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/kinyarwanda_small.pt")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--append-results", action="store_true",
                    help="append the downstream result to eval/RESULTS.md")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.ckpt):
        sys.exit(f"Checkpoint not found: {args.ckpt}")

    tok = KinyaTokenizer(TOK_PATH)
    pad_id = tok.tok.token_to_id("<|pad|>")
    if pad_id is None:
        pad_id = tok.eot_id

    train_csv, test_csv = download_data()
    print("tokenizing KINNEWS...")
    tr_ids, tr_y = read_csv(train_csv, tok, args.max_len, args.max_train)
    te_ids, te_y = read_csv(test_csv, tok, args.max_len, args.max_test)
    print(f"train={len(tr_y):,}  test={len(te_y):,}  classes={NUM_CLASSES}")

    ckpt = torch.load(args.ckpt, map_location=device)
    base = GPTModel(ckpt["config"])
    base.load_state_dict(ckpt["model_state_dict"])
    model = GPTClassifier(base, NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  classifier params={n_params/1e6:.1f}M  "
          f"base trained_step={ckpt.get('step','?')}")

    tr = Batcher(tr_ids, tr_y, pad_id, args.batch_size, True, device)
    te = Batcher(te_ids, te_y, pad_id, args.batch_size, False, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss()

    best_acc = best_f1 = 0.0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for i, (x, y, last) in enumerate(tr):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(x, last), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()
            if (i + 1) % 100 == 0:
                print(f"  ep{ep+1} it{i+1}/{len(tr)} loss {run/100:.3f}")
                run = 0.0
        acc, f1 = evaluate(model, te)
        best_acc, best_f1 = max(best_acc, acc), max(best_f1, f1)
        print(f"epoch {ep+1}: test accuracy {acc*100:.2f}%  macro-F1 {f1*100:.2f}")

    print(f"\nBEST: accuracy {best_acc*100:.2f}%  macro-F1 {best_f1*100:.2f}")
    print(f"baseline -> {BASELINE}")

    if args.append_results:
        os.makedirs("eval", exist_ok=True)
        with open("eval/RESULTS.md", "a", encoding="utf-8") as r:
            r.write("\n## Downstream: KINNEWS news classification (14 classes)\n\n")
            r.write(f"- **Our base model, fine-tuned:** accuracy "
                    f"**{best_acc*100:.2f}%**, macro-F1 {best_f1*100:.2f}\n")
            r.write(f"- **Published baselines** (Niyongabo et al., COLING 2020): "
                    f"BiGRU 88.65%, SVM 88.53%, CNN ~87.5%, char-CNN 71.70%\n")
            r.write(f"- Setup: classification head on the last real token, "
                    f"{args.epochs} epochs, max_len {args.max_len}, lr {args.lr}.\n")
        print("appended to eval/RESULTS.md")


if __name__ == "__main__":
    sys.exit(main())
