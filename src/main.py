import os
import yaml
import yaml
import argparse
import error_analysis
from prepare_data import download_data, prepare_dataset
from train import download_base_model, prepare_lora, train
from inference import inference
from evaluation import load_results, evaluate, summarize
from error_analysis import error_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code completion pipeline")
    parser.add_argument('--config', '-c', required=True, help='Path to train and inference config file')
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Download data
    download_data(config["data_path"])

    # Download model
    base_model, tokenizer = download_base_model(config["base_model_path"])

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
    data = load_results(os.path.join(config["output_dir"], "inference_results.json"))
    scores = evaluate(data)
    summarize(scores, config["output_dir"])
    error_analysis(config["output_dir"])

    # # vLLM inference 
    # # Merge LoRA into base model
    # model = model.merge_and_unload()

    # # Save merged model
    # merged_model_path = os.path.join(config["output_dir"], "merged_model")
    # model.save_pretrained(merged_model_path)
    # print(f"✅ Merged model (LoRA + base) saved to {merged_model_path}")