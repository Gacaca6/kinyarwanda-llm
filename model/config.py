"""
config.py — model + training configuration for the Kinyarwanda LLM.

Architecture is the GPT model from "Build a Large Language Model (From Scratch)"
(Raschka). The values below are adapted for a low-resource language: a smaller
vocab matched to our Kinyarwanda tokenizer, and a small "rwanda-tiny" size you
can actually pretrain on a single modest GPU once you have a few hundred MB+ of
clean text.
"""

# Must match the tokenizer's vocab_size. Phase 3 sweep (16k/32k/50k) on the
# ~178M-token corpus picked 32k as the best fertility/size trade-off
# (1.47 tokens/word, 2.08x vs GPT-2). See eval/tokenizer_sweep.md.
VOCAB_SIZE = 32000

# A small, trainable config to start with. Scale up as your corpus grows.
RWANDA_TINY = {
    "vocab_size": VOCAB_SIZE,
    "context_length": 256,   # Kinyarwanda packs more meaning per token, so a
                             # modest context already covers long passages.
    "emb_dim": 384,
    "n_heads": 6,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# Reference: roughly GPT-2 small shape, for when you have a large corpus + GPU.
RWANDA_SMALL = {
    "vocab_size": VOCAB_SIZE,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

TRAIN = {
    "batch_size": 16,
    "stride": 256,           # = context_length for non-overlapping chunks
    "num_epochs": 10,
    "lr": 4e-4,
    "weight_decay": 0.1,
    "eval_freq": 50,
    "eval_iter": 20,
    "warmup_steps": 100,
}
