import argparse
import inspect
import json


def inspect(results_file):
    # Parse the JSON
    with open(results_file, "r") as inf:
        data = json.load(inf)

    # Iterate and pretty-print prompt and completion
    for entry in data:
        print(f"{'-'*40}\nID: {entry['id']}")
        print("\nPROMPT:\n" + "-" * 40)
        print(entry["prompt"])
        print("\nCOMPLETION:\n" + "-" * 40)
        print(entry["completion"])
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code completion results inspection")
    parser.add_argument(
        "--results", "-r", required=True, help="Path to inference_results.json"
    )
    args = parser.parse_args()
    inspect(args.results)
