#!/usr/bin/env python3
"""
Compute F1 metrics from saved predictions.

Usage:
    python evaluate.py
    python evaluate.py --predictions data/predictions.jsonl --overlap-threshold 0.8
    python evaluate.py --predictions data/predictions.jsonl --show-errors private_phone
    python evaluate.py --compare data/predictions.jsonl data/predictions_finetuned.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_predictions(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_span_set(spans_dict: dict) -> set[tuple]:
    spans: set[tuple] = set()
    for label, offsets in (spans_dict or {}).items():
        for start, end in offsets:
            spans.add((start, end, label))
    return spans


def _overlap_ratio(a: tuple, b: tuple) -> float:
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if intersection == 0:
        return 0.0
    return intersection / min(a[1] - a[0], b[1] - b[0])


def compute_metrics(records: list[dict], overlap_threshold: float = 0.0) -> dict[str, dict]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for rec in records:
        pred_set = to_span_set(rec.get("predicted_spans", {}))
        gold_set = to_span_set(rec.get("gold_spans", {}))

        if overlap_threshold == 0.0:
            for span in pred_set:
                counts[span[2]]["tp" if span in gold_set else "fp"] += 1
            for span in gold_set:
                if span not in pred_set:
                    counts[span[2]]["fn"] += 1
        else:
            unmatched_gold = set(gold_set)
            for pred in pred_set:
                match = next(
                    (g for g in unmatched_gold
                     if g[2] == pred[2] and _overlap_ratio(pred, g) >= overlap_threshold),
                    None,
                )
                if match:
                    counts[pred[2]]["tp"] += 1
                    unmatched_gold.discard(match)
                else:
                    counts[pred[2]]["fp"] += 1
            for span in unmatched_gold:
                counts[span[2]]["fn"] += 1

    metrics: dict[str, dict] = {}
    total_tp = total_fp = total_fn = 0

    for label in sorted(counts):
        tp, fp, fn = counts[label]["tp"], counts[label]["fp"], counts[label]["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        metrics[label] = {"precision": p, "recall": r, "f1": f1, "support": tp + fn}
        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    metrics["OVERALL"] = {"precision": micro_p, "recall": micro_r, "f1": micro_f1, "support": total_tp + total_fn}
    return metrics


def print_metrics(title: str, metrics: dict[str, dict]) -> None:
    width = 62
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")
    print(f"{'Label':<25} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>9}")
    print(f"{'-' * width}")
    for label, m in metrics.items():
        marker = " *" if label == "OVERALL" else ""
        print(f"{label:<25} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f} {m['support']:>9}{marker}")


def print_delta(base: dict[str, dict], ft: dict[str, dict], base_name: str = "Base", ft_name: str = "Fine-tuned") -> None:
    width = 62
    print(f"\n{'=' * width}")
    print(f"  Delta ({ft_name} - {base_name})")
    print(f"{'=' * width}")
    print(f"{'Label':<25} {'DeltaP':>10} {'DeltaR':>10} {'DeltaF1':>10}")
    print(f"{'-' * width}")
    for label in ft:
        if label not in base:
            continue
        dp = ft[label]["precision"] - base[label]["precision"]
        dr = ft[label]["recall"] - base[label]["recall"]
        df = ft[label]["f1"] - base[label]["f1"]
        print(f"{label:<25} {dp:>+10.3f} {dr:>+10.3f} {df:>+10.3f}")


def show_errors(records: list[dict], label: str, n: int = 5) -> None:
    print(f"\n--- {label} errors (up to {n} FP and {n} FN) ---")
    fp_shown = fn_shown = 0
    for rec in records:
        text = rec["text"]
        pred_set = to_span_set(rec.get("predicted_spans", {}))
        gold_set = to_span_set(rec.get("gold_spans", {}))
        for span in pred_set:
            if span[2] == label and span not in gold_set and fp_shown < n:
                print(f"  FP: [{span[0]}:{span[1]}] {text[span[0]:span[1]]!r}")
                fp_shown += 1
        for span in gold_set:
            if span[2] == label and span not in pred_set and fn_shown < n:
                print(f"  FN: [{span[0]}:{span[1]}] {text[span[0]:span[1]]!r}")
                fn_shown += 1
        if fp_shown >= n and fn_shown >= n:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OPF predictions")
    parser.add_argument("--predictions", type=Path, default=Path("data/predictions.jsonl"))
    parser.add_argument("--compare", type=Path, default=None, metavar="FINETUNED_PREDICTIONS",
                        help="Second predictions file to compare against (e.g. fine-tuned model)")
    parser.add_argument("--overlap-threshold", type=float, default=0.8,
                        help="Minimum overlap ratio for TP (0=strict, 0.8=80%% overlap)")
    parser.add_argument("--show-errors", type=str, default=None, metavar="LABEL",
                        help="Show FP/FN examples for a label, e.g. private_phone")
    args = parser.parse_args()

    if not args.predictions.exists():
        raise SystemExit(f"Predictions not found: {args.predictions}\nRun predict.py first.")

    records = load_predictions(args.predictions)
    print(f"Loaded {len(records)} records from {args.predictions}")

    def path_title(p: Path) -> str:
        return f"{p.parent.name}/{p.stem}" if p.parent != Path(".") else p.stem

    metrics = compute_metrics(records, args.overlap_threshold)
    print_metrics(path_title(args.predictions), metrics)

    if args.show_errors:
        show_errors(records, args.show_errors)

    if args.compare:
        if not args.compare.exists():
            raise SystemExit(f"Comparison predictions not found: {args.compare}\nRun predict.py --output {args.compare} first.")
        ft_records = load_predictions(args.compare)
        ft_metrics = compute_metrics(ft_records, args.overlap_threshold)
        print_metrics(path_title(args.compare), ft_metrics)
        print_delta(metrics, ft_metrics, path_title(args.predictions), path_title(args.compare))


if __name__ == "__main__":
    main()
