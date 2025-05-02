import os
import json
from networkx import star_graph
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
import yaml
import argparse
import time
import datasets
from transformers import (Trainer, TrainingArguments, 
                          EarlyStoppingCallback, AutoModelForCausalLM,
                          AutoTokenizer)
from peft import PeftModel
    

def download_data(data_path):
    ##### Data download
    if not os.path.exists(data_path):
        print("Downloading data...")
        dataset = load_dataset("codeparrot/github-code", split="train", streaming=True, trust_remote_code=True)
        python_data = dataset.filter(lambda example: example["language"] == "Python")

        sample = []
        for i, item in tqdm(enumerate(python_data)):
            sample.append(item)
            if i >= 1000:
                break

        with open(data_path, "w") as f:
            json.dump(sample, f, indent=2)

        print(f"Downloaded {len(sample)} Python examples!")
    else:
        print("Data already downloaded.")

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

def prepare_dataset(data_path, tokenizer, output_dir):
    tokenized_path = os.path.join(output_dir, "tokenized_dataset")
    if not os.path.exists(tokenized_path):
        # Load your saved JSON
        with open(data_path, "r") as f:
            samples = json.load(f)

        # Turn into a Huggingface Dataset
        dataset = datasets.Dataset.from_list(samples)

        # Split into train / validation / test
        dataset = dataset.train_test_split(test_size=0.15, seed=42)
        train_val = dataset['train'].train_test_split(test_size=0.1, seed=42)
        train_dataset = train_val['train']
        valid_dataset = train_val['test']
        test_dataset = dataset['test']

        print(f"✅ Dataset split: {len(train_dataset)} train, {len(valid_dataset)} valid, {len(test_dataset)} test")

        # Preprocessing: tokenize everything
        def tokenize_function(example):
            prompt = example.get("code", "")  # Adjust this key if needed!
            inputs = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs

        train_dataset = train_dataset.map(tokenize_function, batched=False)
        valid_dataset = valid_dataset.map(tokenize_function, batched=False)
        test_dataset = test_dataset.map(tokenize_function, batched=False)

        # Save tokenized dataset locally
        os.makedirs(tokenized_path, exist_ok=True)
        train_dataset.save_to_disk(os.path.join(tokenized_path, "train"))
        valid_dataset.save_to_disk(os.path.join(tokenized_path, "valid"))
        test_dataset.save_to_disk(os.path.join(tokenized_path, "test"))

        print(f"✅ Tokenized datasets saved under {tokenized_path}")
    else:
        print("✅ Tokenized dataset already exists. Loading from local files...")

        # Load previously tokenized datasets
        train_dataset = datasets.load_from_disk(os.path.join(tokenized_path, "train"))
        valid_dataset = datasets.load_from_disk(os.path.join(tokenized_path, "valid"))
        test_dataset = datasets.load_from_disk(os.path.join(tokenized_path, "test"))

        print(f"✅ Loaded tokenized datasets: {len(train_dataset)} train, {len(valid_dataset)} valid, {len(test_dataset)} test")

    return train_dataset, valid_dataset, test_dataset

def save_train_summary(trainer, train_config, output_dir, start_time):
    # Create summary dictionary
    summary = {
        "start_time": start_time,
        "end_time": str(time.strftime('%Y-%m-%d %H:%M:%S')),
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
        print(f"✅ LoRA adapters already exist at {lora_output_dir}. Skipping training...")
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
            EarlyStoppingCallback(early_stopping_patience=train_config["early_stopping_patience"])
        ],
    )

    start_time = time.strftime('%Y-%m-%d %H:%M:%S')
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

def inference(config, model, tokenizer, test_dataset):
    # Check if inference results already exist
    output_file = os.path.join(config["output_dir"], "inference_results.json")
    if os.path.exists(output_file):
        print(f"✅ Inference results already exist at {output_file}. Skipping inference...")
    else:

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(device)

        results = []

        print("🚀 Running inference on test dataset...")

        for idx, example in enumerate(test_dataset):
            prompt = example.get("code", "")
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)

            output = model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=True,
                top_p=config["top_p"],
                top_k=config["top_k"],
                temperature=config["temperature"],
            )

            completion = tokenizer.decode(output[0], skip_special_tokens=True)

            results.append({
                "id": idx,
                "prompt": prompt,
                "completion": completion,
            })

            if idx < 3:  # Print only first few completions
                print(f"=== Example {idx} ===")
                print(f"Prompt:\n{prompt}")
                print(f"Completion:\n{completion}")
                print("-" * 50)

        # Save all completions
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✅ All inference results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code completion pipeline")
    parser.add_argument('--config', '-c', required=True, help='Path to train and inference config file')
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Download data
    download_data(config["data_path"])

    # Download model
    base_model, tokenizer = download_base_model(config["model_path"])

    # Prepare LoRA
    model = prepare_lora(base_model, config["lora_path"])

    # Prepare dataset
    train_dataset, valid_dataset, test_dataset = prepare_dataset(
        data_path=config["data_path"],
        tokenizer=tokenizer,
        output_dir=config["output_dir"]
    )

    # Train model
    model = train(model, train_dataset, valid_dataset, config, config["output_dir"])

    # Inference
    inference(config, model, tokenizer, test_dataset)

    # vLLM inference 
    # Merge LoRA into base model
    model = model.merge_and_unload()

    # Save merged model
    merged_model_path = os.path.join(config["output_dir"], "merged_model")
    model.save_pretrained(merged_model_path)
    print(f"✅ Merged model (LoRA + base) saved to {merged_model_path}")