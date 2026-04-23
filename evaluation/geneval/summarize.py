"""Summarize GenEval results.

GenEval's `evaluate_images.py` writes a results.jsonl where each line is something like:
    {"filename": ".../00000/samples/0000.png", "tag": "single_object", "correct": true, ...}

This script groups by `tag`, averages `correct`, and prints a per-tag + overall table.
The output is designed to be paper-ready: match the standard 6-category breakdown.
"""
import argparse
import json
from collections import defaultdict


TAG_ORDER = [
    "single_object", "two_object", "counting",
    "colors", "position", "color_attr",
]


def summarize(results_path: str):
    per_tag = defaultdict(list)
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tag = rec.get("tag", "unknown")
            correct = bool(rec.get("correct", False))
            per_tag[tag].append(correct)

    print(f"{'Tag':<18} {'N':>6} {'Acc (%)':>10}")
    print("-" * 38)
    all_correct = []
    for tag in TAG_ORDER + sorted(k for k in per_tag if k not in TAG_ORDER):
        vals = per_tag.get(tag)
        if not vals:
            continue
        acc = 100 * sum(vals) / len(vals)
        print(f"{tag:<18} {len(vals):>6d} {acc:>9.2f}")
        all_correct.extend(vals)
    print("-" * 38)
    if all_correct:
        overall = 100 * sum(all_correct) / len(all_correct)
        print(f"{'OVERALL':<18} {len(all_correct):>6d} {overall:>9.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("results_path", help="GenEval's results.jsonl")
    args = p.parse_args()
    summarize(args.results_path)
