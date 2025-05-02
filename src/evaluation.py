import json
import argparse
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

def evaluate(results):
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    scores = []
    for item in results:
        ref = item.get("reference", "").strip()
        hyp = item.get("completion", "").strip()

        if not ref:
            continue

        # BLEU
        bleu = sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth)

        # ROUGE-L
        rouge_l = rouge.score(ref, hyp)["rougeL"].fmeasure

        scores.append({
            "id": item["id"],
            "bleu": bleu,
            "rougeL": rouge_l,
        })

    return scores

def summarize(scores, output_dir):
    bleu_avg = sum(s["bleu"] for s in scores) / len(scores)
    rouge_avg = sum(s["rougeL"] for s in scores) / len(scores)

    print(f"\n=== Evaluation Summary ===")
    print(f"Avg BLEU:    {bleu_avg:.4f}")
    print(f"Avg ROUGE-L: {rouge_avg:.4f}")
    print(f"Examples evaluated: {len(scores)}")

    # Save per-sample scores
    df = pd.DataFrame(scores)
    csv_path = f"{output_dir}/evaluation_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Evaluation scores logged to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", "-r", required=True, help="Path to inference_results.json")
    parser.add_argument("--output_dir", "-o", required=False, default="analysis_output", help="Where to save evaluation CSV")
    args = parser.parse_args()

    results = load_results(args.results)
    scores = evaluate(results)
    summarize(scores, args.output_dir)