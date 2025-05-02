from transformers import AutoTokenizer
from datasets import load_dataset
import argparse
import json
import os
import ast
import datasets


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

def extract_prompt_and_completion(code):
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Build function signature
                sig = f"def {node.name}(" + ", ".join(arg.arg for arg in node.args.args) + "):"
                lines = code.strip().split("\n")
                body_lines = lines[1:] if lines[0].strip().startswith(sig) else lines
                return sig, "\n".join(body_lines).strip()
    except Exception:
        pass

    # Fallback: split first line vs rest
    lines = code.strip().split("\n")
    if lines and lines[0].startswith("def "):
        return lines[0], "\n".join(lines[1:]).strip()
    return None, None

def prepare_dataset(data_path, tokenizer, output_dir):
    tokenized_path = os.path.join(output_dir, "tokenized_dataset")
    if not os.path.exists(tokenized_path):
        # Load your saved JSON
        with open(data_path, "r") as f:
            samples = json.load(f)

        # Pre-process to add prompt and completion fields
        cleaned_samples = []
        for sample in samples:
            code = sample.get("code", "")
            prompt, completion = extract_prompt_and_completion(code)
            if prompt and completion:
                cleaned_samples.append({
                    "prompt": prompt,
                    "completion": completion,
                    "code": code,  # optional: keep original
                    **{k: v for k, v in sample.items() if k != "code"}
                })

        print(f"✅ Kept {len(cleaned_samples)} samples with valid prompt/completion split")

        # Turn into a Huggingface Dataset
        dataset = datasets.Dataset.from_list(cleaned_samples)

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