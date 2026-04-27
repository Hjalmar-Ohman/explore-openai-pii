"""
Run OPF inference on Modal GPU.

Usage:
    modal run modal_predict.py
    modal run modal_predict.py --test-data data/test.jsonl --output data/predictions.jsonl
"""

import modal
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "pydantic>=2.0.0",
        "git+https://github.com/openai/privacy-filter.git",
    )
    .env({"OPF_MOE_TRITON": "0"})
)

app = modal.App("opf-predict", image=image)

@app.function(gpu="T4", timeout=3600)
def predict(records_jsonl: str) -> str:
    import json
    import torch
    from opf._api import OPF

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    records = [json.loads(l) for l in records_jsonl.splitlines() if l.strip()]
    print(f"Loaded {len(records)} records")

    opf = OPF(device=device)
    results = []
    for i, record in enumerate(records):
        result = opf.redact(record["text"])
        predicted: dict[str, list[list[int]]] = {}
        for span in result.detected_spans:
            predicted.setdefault(span.label, []).append([span.start, span.end])
        results.append({
            "text": record["text"],
            "gold_spans": record.get("spans", {}),
            "predicted_spans": predicted,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(records)}")

    print(f"\nDone — {len(results)} records")
    return "\n".join(json.dumps(r) for r in results)


@app.local_entrypoint()
def main(test_data: str = "data/test.jsonl", output: str = "data/predictions.jsonl"):
    test_path = Path(test_data)
    if not test_path.exists():
        raise SystemExit(f"Test data not found: {test_path}\nRun generate_data.py first.")

    result = predict.remote(test_path.read_text())

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result)
    print(f"Saved predictions to {out_path}")
