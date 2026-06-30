"""
generate.py — load a trained Kinyarwanda GPT and generate text.

    python -m scripts.generate "Abanyarwanda"
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.gpt import GPTModel, generate                 # noqa: E402
from model.data_loader import KinyaTokenizer             # noqa: E402

CKPT = "checkpoints/kinyarwanda_gpt.pt"
TOK_PATH = "tokenizer/kinyarwanda_bpe/tokenizer.json"


def main(prompt, max_new_tokens=40):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = KinyaTokenizer(TOK_PATH)
    ckpt = torch.load(CKPT, map_location=device)
    model = GPTModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    out = generate(model, ids, max_new_tokens,
                   ckpt["config"]["context_length"], temperature=0.8, top_k=40,
                   eos_id=tokenizer.eot_id)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "U Rwanda"
    main(prompt)
