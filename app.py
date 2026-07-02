"""
app.py — minimal Gradio demo for the Kinyarwanda base model (Phase 7).

Type a Kinyarwanda prompt; the model continues it. This is a *base* (text-completion)
model, not an instruction assistant — it continues text, it does not answer questions.

Run locally:
  pip install gradio
  KIN_CKPT=checkpoints/kinyarwanda_small.pt python app.py

On Hugging Face Spaces: put this file as app.py, add the tokenizer/ + model/ dirs and
a requirements.txt (gradio, torch, tokenizers, numpy), and point KIN_CKPT at the
weights (upload them to the Space or load from a HF model repo).
"""
import os
import torch

from model.gpt import GPTModel, generate
from model.data_loader import KinyaTokenizer

CKPT = os.environ.get("KIN_CKPT", "checkpoints/kinyarwanda_small.pt")
TOK_PATH = "tokenizer/kinyarwanda_bpe/tokenizer.json"

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tok = KinyaTokenizer(TOK_PATH)
_ckpt = torch.load(CKPT, map_location=_device)
_model = GPTModel(_ckpt["config"]).to(_device)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()
_ctx = _ckpt["config"]["context_length"]


def complete(prompt, max_new_tokens=60, temperature=0.8, top_k=40):
    if not prompt or not prompt.strip():
        return ""
    ids = torch.tensor([_tok.encode(prompt)], device=_device)
    out = generate(_model, ids, int(max_new_tokens), _ctx,
                   temperature=float(temperature), top_k=int(top_k),
                   eos_id=_tok.eot_id)
    return _tok.decode(out[0].tolist())


EXAMPLES = [
    ["U Rwanda", 60, 0.8, 40],
    ["Ikinyarwanda ni ururimi", 60, 0.8, 40],
    ["Uburezi ni", 60, 0.8, 40],
    ["Mu mujyi wa Kigali", 60, 0.8, 40],
    ["Umuco nyarwanda", 60, 0.8, 40],
]


def build_demo():
    import gradio as gr
    return gr.Interface(
        fn=complete,
        inputs=[
            gr.Textbox(label="Prompt (Kinyarwanda)", value="U Rwanda", lines=2),
            gr.Slider(10, 150, value=60, step=10, label="Max new tokens"),
            gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature"),
            gr.Slider(1, 100, value=40, step=1, label="Top-k"),
        ],
        outputs=gr.Textbox(label="Continuation", lines=6),
        title="Kinyarwanda LLM — base model (v0)",
        description=("A from-scratch GPT for Kinyarwanda (134M params, 32k Kinyarwanda "
                     "BPE, trained on ~174M tokens). This is a **base** model: it "
                     "*continues* Kinyarwanda text, it does not follow instructions or "
                     "answer questions. It can be fluent but also repeat or make things "
                     "up — do not rely on it for facts."),
        examples=EXAMPLES,
        flagging_mode="never",   # gradio >= 5 (was allow_flagging in gradio 4)
    )


if __name__ == "__main__":
    print(f"loaded {sum(p.numel() for p in _model.parameters())/1e6:.1f}M params "
          f"on {_device} from {CKPT}")
    build_demo().launch()
