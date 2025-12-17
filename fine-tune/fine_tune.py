"""
Fine-tuning script for google/flan-t5-large model.
This script provides a flexible fine-tuning setup with sample data.
Customize the dataset and training parameters as needed.
"""
import os
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
)
import argparse
from datasets import Dataset
import torch

def load_custom_dataset(dataset_path):
    """
    Load a custom dataset from a JSON file.
    Expected format: [{"input": "...", "output": "..."}, ...]
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"Loading custom dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return Dataset.from_list(data)


def detect_model_kind(model_path, override_kind=None):
    """Detect whether the model is seq2seq (encoder-decoder) or causal (decoder-only).
    Returns 'seq2seq' or 'causal'."""
    if override_kind in ("seq2seq", "causal"):
        return override_kind
    try:
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        mt = getattr(config, "model_type", "")
        # common seq2seq types
        seq2seq_types = {"t5", "mt5", "bart", "pegasus", "marian", "led", "mbart", "flan"}
        if mt and any(s in mt for s in seq2seq_types) or mt in seq2seq_types:
            return "seq2seq"
        return "causal"
    except Exception:
        # fallback to seq2seq
        return "seq2seq"

def preprocess_seq2seq(examples, tokenizer, max_input_length=512, max_target_length=128):
    model_inputs = tokenizer(
        examples["input"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        examples["output"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_causal(example, tokenizer, max_length=1024, max_input_length=512):
    # single example (not batched)
    input_text = example.get("input", "")
    target_text = example.get("output", "")
    # tokenize input only to compute input length in tokens
    input_ids = tokenizer(input_text, add_special_tokens=False).input_ids
    # combine prompt and target with eos token
    eos = tokenizer.eos_token or tokenizer.sep_token or ""
    combined = input_text + (" " + eos if eos else "") + " " + target_text
    encoded = tokenizer(
        combined,
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    labels = encoded["input_ids"].copy()
    # mask loss for prompt tokens
    for i in range(len(input_ids)):
        if i < len(labels):
            labels[i] = -100
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded.get("attention_mask", None), "labels": labels}

def fine_tune_model():
    """Fine-tune the FLAN-T5 model."""
    # Paths
    model_path = "/models/flan-t5-large"
    output_path = "/models/flan-t5-large-finetuned"
    dataset_path = "/data/training_data.json"
    
    print("=" * 50)
    print("Starting FLAN-T5-Large Fine-tuning")
    print("=" * 50)
    
    # Load tokenizer and model
    # parse optional CLI args / env
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", model_path))
    parser.add_argument("--output_path", default=os.environ.get("OUTPUT_PATH", output_path))
    parser.add_argument("--dataset_path", default=os.environ.get("DATASET_PATH", dataset_path))
    parser.add_argument("--model_kind", default=os.environ.get("MODEL_KIND", None), help="Override model kind: seq2seq or causal")
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 3)))
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 4)))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", 5e-5)))
    args, _ = parser.parse_known_args()
    model_path = args.model_path
    output_path = args.output_path
    dataset_path = args.dataset_path

    print("\nLoading tokenizer and model (local files only)...")
    # prefer local files only to avoid network downloads inside the container
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_kind = detect_model_kind(model_path, override_kind=args.model_kind)
    print(f"Detected model kind: {model_kind}")
    if model_kind == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

    print(f"Model loaded from: {model_path}")
    try:
        # `num_parameters` helper exists on HF models; wrap for safety across versions
        nparams = model.num_parameters() if hasattr(model, 'num_parameters') else sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {nparams:,}")
    except Exception:
        pass
    
    # Load and prepare dataset
    print("\nPreparing dataset...")
    dataset = load_custom_dataset(dataset_path)
    print(f"Dataset size: {len(dataset)} examples")
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    # normalize dataset fields: if data has 'note' and 'output' (list), convert to input/output strings
    def normalize(example):
        ex = dict(example)
        if "input" not in ex:
            if "note" in ex:
                ex["input"] = ex.pop("note")
        # if output is a list/dict, serialize to JSON string
        if "output" in ex and not isinstance(ex["output"], str):
            ex["output"] = json.dumps(ex["output"], ensure_ascii=False)
        return ex

    dataset = dataset.map(lambda x: normalize(x), batched=False)

    if model_kind == "seq2seq":
        tokenized_dataset = dataset.map(
            lambda x: preprocess_seq2seq(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names,
        )
    else:
        # causal: process per-example
        tokenized = dataset.map(lambda x: preprocess_causal(x, tokenizer), batched=False)
        # tokenized already has input_ids, attention_mask, labels
        tokenized_dataset = tokenized
    
    # Split into train and validation (90/10 split)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    
    # Data collator
    # Data collator depending on model type
    if model_kind == "seq2seq":
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
        )
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    
    # Training arguments
    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_path}/logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        learning_rate=args.lr,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
    )
    
    # Create Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")
    trainer.train()
    
    # Save the fine-tuned model
    print("\n" + "=" * 50)
    print("Saving fine-tuned model...")
    print("=" * 50)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\nFine-tuned model saved to: {output_path}")
    print("Fine-tuning complete!")

if __name__ == "__main__":
    fine_tune_model()
