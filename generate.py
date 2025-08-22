from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch, json

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "checkpoints/op-next"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(model, ADAPTER)

def chat(prompt, max_new_tokens=512, temperature=0.7):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, pad_token_id=tok.eos_token_id
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

context = "- Ch.1155 The Rocks Pirates: Levely incident; Hachinosu plan\n- Ch.1156 Idols: Kuja 'idol' era; Pirate Island consolidated"
prompt = (
    "You are a One Piece story forecaster. Using only the prior chapter context, "
    "predict the core beats of the next chapter in 3–7 bullets. Keep them short.\n\n"
    f"### Prior context (last 2):\n{context}\n\n### Task: Predict beats for Chapter 1157\n"
)
print(chat(prompt))
