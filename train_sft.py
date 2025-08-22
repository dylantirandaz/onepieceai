import json, os
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import torch
from transformers import BitsAndBytesConfig
from inspect import signature

try:
    from trl import SFTConfig
    HAS_SFTCONFIG = True
except Exception:
    HAS_SFTCONFIG = False

CUDA = torch.cuda.is_available()
BF16 = CUDA and torch.cuda.is_bf16_supported()
FP16 = CUDA and not BF16
DTYPE = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  
TRAIN_FILE = "out_train/multitask_mix.train.jsonl"
VAL_FILE   = "out_train/multitask_mix.val.jsonl"
OUTPUT_DIR = "checkpoints/op-next"
MAX_LEN = 512        
BATCH_SIZE = 4       
GRAD_ACCUM = 2       
LR = 2e-4
EPOCHS = 1           

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def ds_from_jsonl(path):
    return load_dataset("json", data_files=path, split="train")

train_ds = ds_from_jsonl(TRAIN_FILE)
val_ds   = ds_from_jsonl(VAL_FILE)

def format_row(ex):
    return f"{ex['prompt'].strip()}\n\n### ANSWER\n{ex['completion'].strip()}"

train_ds = train_ds.map(lambda ex: {"text": format_row(ex)})
val_ds   = val_ds.map(lambda ex: {"text": format_row(ex)})

bnb_config = (
    BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
    )
    if CUDA
    else None
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto" if CUDA else "cpu",
    quantization_config=bnb_config,
    torch_dtype=DTYPE,
)

peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.05,  #fewer trainable params
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, peft_cfg)

if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()

eval_kw = (
    {"eval_strategy": "steps"}
    if "eval_strategy" in signature(TrainingArguments.__init__).parameters
    else {"evaluation_strategy": "steps"}
)

if HAS_SFTCONFIG:
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        bf16=BF16,
        fp16=FP16,
        optim="paged_adamw_8bit" if CUDA else "adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        max_grad_norm=1.0,
        ##sft-specific
        dataset_text_field="text",
        max_length=MAX_LEN,
        packing=False,
        dataloader_num_workers=4,
        evaluation_strategy="no",  
        save_steps=0,              
        logging_steps=10,
        **eval_kw,
    )
else:
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        bf16=BF16,
        fp16=FP16,
        optim="paged_adamw_8bit" if CUDA else "adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        max_grad_norm=1.0,
        **eval_kw,
    )

sft_kwargs = {}
sft_params = signature(SFTTrainer.__init__).parameters
if "processing_class" in sft_params:
    sft_kwargs["processing_class"] = tokenizer
elif "tokenizer" in sft_params:
    sft_kwargs["tokenizer"] = tokenizer

trainer_kwargs = dict(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=args,
    **sft_kwargs,
)

if not HAS_SFTCONFIG:
    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = MAX_LEN
    if "packing" in sft_params:
        trainer_kwargs["packing"] = False
    if "formatting_func" in sft_params:
        def _fmt(ex): return ex["text"]
        trainer_kwargs["formatting_func"] = _fmt
    elif "dataset_text_field" in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"

trainer = SFTTrainer(**trainer_kwargs)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done. Adapter saved to:", OUTPUT_DIR)
