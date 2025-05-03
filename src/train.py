import json
import os
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def download_base_model(base_model_path):
    ##### Model download
    base_model_name = "codellama/CodeLlama-7b-Instruct-hf"

    if not os.path.exists(base_model_path):
        print("Downloading model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Save locally
        base_model.save_pretrained(base_model_path)
        tokenizer.save_pretrained(base_model_path)
    else:
        print("Loading model from local path...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Found the {device} device, saving the model.")
    # base_model = base_model.to(device)

    tokenizer.pad_token = tokenizer.eos_token

    return base_model, tokenizer


def prepare_lora(base_model, lora_path):
    if not os.path.exists(lora_path):
        print("🚀 No LoRA adapters found. Preparing fresh LoRA configuration...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],  # Depends on model
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(base_model, lora_config)
    else:
        print(f"✅ Found existing LoRA adapters at {lora_path}, loading them...")
        model = PeftModel.from_pretrained(base_model, lora_path)

    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    return model


def save_train_summary(trainer, train_config, output_dir, start_time):
    # Create summary dictionary
    summary = {
        "start_time": start_time,
        "end_time": str(time.strftime("%Y-%m-%d %H:%M:%S")),
        "total_steps": trainer.state.global_step,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "train_config": train_config,  # snapshot of the config used
    }

    # Save to JSON
    summary_path = os.path.join(output_dir, "train_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Training summary saved to {summary_path}")


def train(model, train_dataset, valid_dataset, train_config, output_dir):
    # Check if LoRA adapters already exist
    lora_output_dir = os.path.join(output_dir, "lora_adapters")
    if os.path.exists(os.path.join(lora_output_dir, "adapter_model.safetensors")):
        print(
            f"✅ LoRA adapters already exist at {lora_output_dir}. Skipping training..."
        )
        model = PeftModel.from_pretrained(model, lora_output_dir)
        return model

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        num_train_epochs=train_config["num_train_epochs"],
        learning_rate=train_config["learning_rate"],
        logging_steps=train_config["logging_steps"],
        save_strategy=train_config["save_strategy"],
        save_total_limit=train_config["save_total_limit"],
        eval_strategy=train_config["eval_strategy"],
        eval_steps=train_config["eval_steps"],
        bf16=False,  # no BF16 on Mac M1
        fp16=False,  # safer off for M1
        optim="adamw_torch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=train_config["early_stopping_patience"]
            )
        ],
    )

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    trainer.train()

    # Save training logs (losses, metrics) to CSV
    logs = trainer.state.log_history  # list of dicts

    import pandas as pd

    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(os.path.join(output_dir, "train_logs.csv"), index=False)

    print(f"✅ Training logs saved to {os.path.join(output_dir, 'train_logs.csv')}")

    # Save LoRA adapters
    lora_output_dir = os.path.join(output_dir, "lora_adapters")
    model.save_pretrained(lora_output_dir)
    print(f"✅ LoRA adapters saved to {lora_output_dir}")

    # Save summary JSON
    save_train_summary(trainer, train_config, output_dir, start_time)

    return model
