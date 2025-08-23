import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "checkpoints/op-next"

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(ADAPTER, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

quant_cfg = None
base_kwargs = {}
if device == "cuda":
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    base_kwargs.update(dict(
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ))
else:
    base_kwargs.update(dict(
        device_map=None,
        torch_dtype=torch.float32,
    ))

base = AutoModelForCausalLM.from_pretrained(BASE, **base_kwargs)
model = PeftModel.from_pretrained(base, ADAPTER)
model.to(device)
model.eval()

@torch.inference_mode()
def chat(prompt: str, max_new_tokens=256, temperature=0.7, top_p=0.9):
    model_input = prompt 
    inputs = tok(model_input, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    new_tokens = gen[0, input_len:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

if __name__ == "__main__":
    context = "- Ch.1155 The Rocks Pirates: Levely incident; Hachinosu plan\n- Ch.1156 Idols: Kuja 'idol' era; Pirate Island consolidated"
    prompt = (
        "You are a One Piece story forecaster. Using only the prior chapter context, "
        "predict the core beats of the next chapter in 3–7 bullets. Keep them short.\n\n"
        f"### Prior context (last 2):\n{context}\n\n### Task: Predict beats for Chapter 1157\n"
    )
    print(chat(prompt))
