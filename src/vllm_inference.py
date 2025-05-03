import torch
from vllm import LLM, SamplingParams


def vllm_inference(config, merged_model_path, test_dataset):
    # Instead of tokenizer/model from HuggingFace, use vLLM LLM
    llm = LLM(model=merged_model_path)
    sampling_params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        max_tokens=config["max_new_tokens"],
        n=1,
    )

    results = []

    for idx, example in enumerate(test_dataset):
        prompt = example.get("code", "")

        # Generate completion using vLLM
        outputs = llm.generate([prompt], sampling_params)
        completion = outputs[0].outputs[0].text  # Get first generated output

        results.append(
            {
                "id": idx,
                "prompt": prompt,
                "completion": completion,
            }
        )

        if idx < 3:
            print(f"=== Example {idx} ===")
            print(f"Prompt:\n{prompt}")
            print(f"Completion:\n{completion}")
            print("-" * 50)
