"""
validate_pipeline.py — verify the data pipeline WITHOUT torch.
Confirms (1) the tokenizer round-trips Kinyarwanda, (2) sliding-window chunks
have correct shape and the next-token offset (target = input shifted by 1),
and (3) reports the parameter count the model WILL have, computed analytically.
"""
import json
from tokenizers import Tokenizer

TOK = "tokenizer/kinyarwanda_bpe/tokenizer.json"
CORPUS = "data/kinyarwanda_corpus.txt"
CFG = {"vocab_size": 8000, "context_length": 256, "emb_dim": 384,
       "n_heads": 6, "n_layers": 6}


def chunk(ids, max_len, stride):
    pairs = []
    for i in range(0, len(ids) - max_len, stride):
        pairs.append((ids[i:i + max_len], ids[i + 1:i + max_len + 1]))
    return pairs


def param_count(c):
    v, d, L, ctx = c["vocab_size"], c["emb_dim"], c["n_layers"], c["context_length"]
    emb = v * d + ctx * d
    per_layer = 4 * (d * d) + 2 * (d * 4 * d) + 4 * d  # attn + ffn + 2 LayerNorms
    head = v * d
    return emb + L * per_layer + head


def main():
    tok = Tokenizer.from_file(TOK)
    text = open(CORPUS, encoding="utf-8").read()

    s = "Abanyarwanda bishimiye uburezi."
    ids = tok.encode(s).ids
    print("round-trip:", repr(tok.decode(ids)), "| ids:", ids[:8], "...")

    all_ids = tok.encode(text).ids
    pairs = chunk(all_ids, CFG["context_length"], CFG["context_length"])
    inp, tgt = pairs[0]
    assert len(inp) == len(tgt) == CFG["context_length"]
    assert inp[1:] == tgt[:-1], "target must be input shifted by one"
    print(f"corpus tokens: {len(all_ids):,}  -> training pairs: {len(pairs):,} "
          f"(each {CFG['context_length']} tokens)")
    print("offset check (target == input shifted by 1): PASS")
    print(f"model parameter count (analytic): {param_count(CFG)/1e6:.1f}M")


if __name__ == "__main__":
    main()
