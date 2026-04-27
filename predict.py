#!/usr/bin/env python3
"""
Run inference with openai/privacy-filter and save predictions to disk.

Usage:
    OPF_MOE_TRITON=0 python predict.py
    OPF_MOE_TRITON=0 python predict.py --model finetuned_model --output data/predictions_finetuned.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import torch
from pathlib import Path

os.environ.setdefault("OPF_MOE_TRITON", "0")

from opf._api import OPF


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OPF inference and save predictions")
    parser.add_argument("--test-data", type=Path, default=Path("data/test.jsonl"))
    parser.add_argument("--model", type=str, default=None,
                        help="Checkpoint path (default: base openai/privacy-filter)")
    parser.add_argument("--output", type=Path, default=Path("data/predictions.jsonl"))
    args = parser.parse_args()

    if not args.test_data.exists():
        raise SystemExit(f"Test data not found: {args.test_data}\nRun generate_data.py first.")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    records = []
    with open(args.test_data) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {args.test_data}")

    model_label = args.model or "openai/privacy-filter (base)"
    print(f"Loading model: {model_label}")
    opf = OPF(model=args.model, device=device)

    print("Running inference...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for i, record in enumerate(records):
            result = opf.redact(record["text"])
            predicted: dict[str, list[list[int]]] = {}
            for span in result.detected_spans:
                predicted.setdefault(span.label, []).append([span.start, span.end])
            out.write(json.dumps({
                "text": record["text"],
                "gold_spans": record.get("spans", {}),
                "predicted_spans": predicted,
            }) + "\n")
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(records)}")

    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
