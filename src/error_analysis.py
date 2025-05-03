import json
import os
import time

import boto3
import botocore
import pandas as pd
from tqdm import tqdm

session = boto3.Session(profile_name="ogokmen_bedrock")

bedrock = session.client("bedrock-runtime", region_name="us-east-1")


def error_analysis(output_dir):

    def build_prompt(entry):
        return f"""
    PROMPT:
    {entry['prompt']}

    COMPLETION:
    {entry['completion']}

    REFERENCE:
    {entry['reference']}

    TAG OPTIONS: [correct, wrong_logic, syntax_error, partial, other]

    Q: How would you tag this completion based on how well it matches the reference? Only return the tag.
    A:
    """

    def tag_with_bedrock(
        prompt, model_id="anthropic.claude-3-sonnet-20240229-v1:0", max_retries=5
    ):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 20,
            "temperature": 0.0,
            "top_p": 1.0,
        }
        for attempt in range(max_retries):
            try:
                response = bedrock.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                result = json.loads(response["body"].read())
                tag = result["content"][0]["text"].strip().lower()
                return tag

            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ThrottlingException":
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s, 8s, ...
                    print(
                        f"⏳ Throttled. Retrying in {wait_time} seconds (attempt {attempt+1}/{max_retries})..."
                    )
                    time.sleep(wait_time)
                else:
                    print("⚠️ AWS ClientError:", e)
                    break

            except Exception as e:
                print("⚠️ General error:", e)
                break

        return "error"

    # Load inference results
    with open(os.path.join(output_dir, "inference_results.json")) as f:
        data = json.load(f)

    tagged = []
    for entry in tqdm(data):
        prompt = build_prompt(entry)
        tag = tag_with_bedrock(prompt)
        tagged.append(
            {
                "id": entry["id"],
                "prompt": entry["prompt"],
                "completion": entry["completion"],
                "reference": entry["reference"],
                "tag": tag,
            }
        )
        time.sleep(5.0)  # To be kind to the API

    # Save to CSV
    df = pd.DataFrame(tagged)
    df.to_csv(os.path.join(output_dir, "annotated_results.csv"), index=False)
    print(f"✅ Tagged results saved to {output_dir}/annotated_results.csv")
