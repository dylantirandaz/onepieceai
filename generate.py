import json, argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "checkpoints/op-next"
CHAPTERS_JSONL = "out/onepiece_chapters.jsonl"  
MAX_INPUT_TOKENS = 1400   
DEFAULT_K = None          
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

FORECAST_SYSTEM = "You forecast concise bullet beats for the next One Piece chapter. Only plausible, spoiler-free predictions."

def load_chapters(path: str):
    rows = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Chapters file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line)
                if isinstance(obj.get("chapter_id"), int):
                    rows.append(obj)
            except:
                pass
    rows.sort(key=lambda r: r["chapter_id"])
    return rows

def chapter_line(r):
    cid = r["chapter_id"]
    title = (r.get("title_en") or f"Chapter {cid}").strip()
    summ = (r.get("summary_short_paraphrase") or r.get("summary_short") or "")[:300].replace("\n"," ")
    return f"- Ch.{cid} {title}: {summ}"

def build_context_block(chapters, target_chapter: int, k: int | None, max_tokens: int):
    prev = [r for r in chapters if r["chapter_id"] < target_chapter]
    if not prev:
        raise ValueError("No previous chapters available for provided target.")
    if k is not None:
        prev = prev[-k:]
    lines = [chapter_line(r) for r in prev]
    ctx_text = "\n".join(lines)
    tokens = tok.encode(ctx_text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return ctx_text, len(tokens), len(lines)
    acc = []
    total_tokens = 0
    for line in reversed(lines):
        t = tok.encode(line+"\n", add_special_tokens=False)
        if total_tokens + len(t) > max_tokens:
            break
        acc.append(line)
        total_tokens += len(t)
    acc = list(reversed(acc))
    return "\n".join(acc), total_tokens, len(acc)

def build_prompt(context_text: str, next_ch_number: int) -> str:
    user_block = (
        f"{FORECAST_SYSTEM}\n\n"
        f"Given the previous chapters:\n{context_text.strip()}\n\n"
        f"Predict 3-7 concise bullet points for Chapter {next_ch_number}.\n"
        "Bullets start with '-'.",
    )
    return f"{user_block}\n\n### ANSWER\n"

@torch.inference_mode()
def forecast(context_text: str, next_ch_number: int,
             max_new_tokens=160,
             temperature=0.6,
             top_p=0.9,
             top_k=50,
             repetition_penalty=1.15):
    prompt = build_prompt(context_text, next_ch_number)
    inputs = tok(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    new_tokens = gen[0, input_len:]
    text = tok.decode(new_tokens, skip_special_tokens=True)
    for stop_token in ["\n\n#", "\n###", "\n\nEND"]:
        idx = text.find(stop_token)
        if idx != -1:
            text = text[:idx]
            break
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("-")]
    return "\n".join(lines).strip() or text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapters_jsonl", default=CHAPTERS_JSONL, help="Path to chapters JSONL")
    ap.add_argument("--chapter", type=int, required=True, help="Target chapter to forecast")
    ap.add_argument("--k", type=int, default=DEFAULT_K if DEFAULT_K is not None else -1,
                    help="Last k chapters (set -1 to use all within token cap)")
    ap.add_argument("--max_ctx_tokens", type=int, default=MAX_INPUT_TOKENS)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    args = ap.parse_args()

    chapters = load_chapters(args.chapters_jsonl)
    k = None if args.k == -1 else args.k
    ctx, used_tokens, used_ch_count = build_context_block(
        chapters, args.chapter, k, args.max_ctx_tokens
    )
    print(f"[Context] chapters used: {used_ch_count} tokens: {used_tokens}")
    output = forecast(ctx, args.chapter, max_new_tokens=args.max_new_tokens)
    print("=== Forecast ===")
    print(output)

if __name__ == "__main__":
    main()
