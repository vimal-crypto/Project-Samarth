# train_reasoner_lora.py — LoRA fine-tune flan-t5-base on train_qa.jsonl
import os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "google/flan-t5-base"
OUT_DIR = "samarth_reasoner"
TRAIN_FILE = "train_qa.jsonl"
MAX_SAMPLES = None  # set e.g. 1000 for quick run

def format_example(ex):
    # concatenating instruction + context → target = output
    prompt = f"Instruction: {ex['instruction']}\nContext: {ex['context']}\nAnswer:"
    return {"input_text": prompt, "labels": ex["output"]}

def main():
    ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
    if MAX_SAMPLES:
        ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    ds = ds.map(format_example)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q","v","k","o"],  # safe defaults for T5
        bias="none"
    )
    model = get_peft_model(model, peft_cfg)

    def tokenize_fn(batch):
        model_inputs = tok(batch["input_text"], max_length=512, truncation=True)
        labels = tok(text_target=batch["labels"], max_length=256, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
    collator = DataCollatorForSeq2Seq(tokenizer=tok, model=model)

    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=True,
        logging_steps=25,
        save_strategy="epoch",
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        tokenizer=tok
    )
    trainer.train()
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"[OK] Saved LoRA adapter to {OUT_DIR}")

if __name__ == "__main__":
    main()
