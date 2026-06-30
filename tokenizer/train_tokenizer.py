"""
train_tokenizer.py
Trains a byte-level BPE tokenizer on Kinyarwanda text.

WHY THIS IS THE KEY ADAPTATION "FOR RWANDANS"
---------------------------------------------
The book (LLMs-from-scratch) uses tiktoken's "gpt2" tokenizer, whose merges
were learned on English. Kinyarwanda is a Bantu, agglutinative language: a
single word packs subject/object markers, tense, aspect and a stem
(e.g. "ntibazabikora" ~ "they will not do it"). An English-trained tokenizer
shatters such words into many tokens, which wastes context and makes learning
harder. Research on Kinyarwanda (KinyaBERT, Nzeyimana & Niyongabo Rubungo,
ACL 2022) shows tokenization is the critical bottleneck for the language.

Training the tokenizer on Kinyarwanda itself is the cheapest, highest-impact
first step. (A future upgrade is a morphology-aware tokenizer — see README.)
"""
import argparse
from tokenizers import ByteLevelBPETokenizer

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]


def train(corpus_path, out_dir, vocab_size):
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[corpus_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    import os
    os.makedirs(out_dir, exist_ok=True)
    tok.save_model(out_dir)        # vocab.json + merges.txt
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    print(f"Saved Kinyarwanda tokenizer (vocab={vocab_size}) -> {out_dir}")
    return tok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="../data/kinyarwanda_corpus.txt")
    p.add_argument("--out", default="kinyarwanda_bpe")
    p.add_argument("--vocab_size", type=int, default=8000)
    a = p.parse_args()
    train(a.corpus, a.out, a.vocab_size)
