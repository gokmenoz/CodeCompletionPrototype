import os
import json
import torch


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
                "reference": example.get("completion", "")
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