# explore-openai-pii

Evaluate OpenAI's `privacy-filter` model: generate synthetic PII data with NVIDIA DataDesigner, fine-tune on it, and compare F1 against the base model.

## Pipeline

```
generate_data.py  →  opf train  →  evaluate.py
     ↓                   ↓              ↓
 data/train.jsonl   finetuned_model  F1 comparison table
 data/test.jsonl
```

## Setup

**1. Install all dependencies (including the `opf` CLI):**

```bash
pip install -r requirements.txt
# Verify: opf --help
```

**3. Set your API key:**

```bash
export OPENAI_API_KEY=sk-...
```

## Run the full pipeline

```bash
bash pipeline.sh
```

Or step by step:

```bash
# Step 1 — generate 500 synthetic records, split 80/20 into data/train.jsonl + data/test.jsonl
python generate_data.py --num-records 500 --output-dir data/

# Step 2 — fine-tune on training split (saves checkpoint to finetuned_model/)
opf train data/train.jsonl --output-dir finetuned_model

# Step 3 — base model quick eval via opf CLI
opf eval data/test.jsonl

# Step 4 — Python F1 comparison: base openai/privacy-filter vs fine-tuned checkpoint
python evaluate.py
```

## Output

`evaluate.py` prints per-label and micro-averaged precision / recall / F1 for both models, then a delta table:

```
============================================================
  Base model: openai/privacy-filter
============================================================
Label                     Precision     Recall       F1   Support
------------------------------------------------------------
account_number               0.921      0.934    0.927        86 
private_date                 0.944      0.961    0.952       120
private_email                0.987      0.993    0.990       118
...
OVERALL                      0.953      0.968    0.960       712 *

============================================================
  Fine-tuned: finetuned_model
============================================================
...

============================================================
  Delta (Fine-tuned − Base)
============================================================
Label                    ΔPrecision    ΔRecall      ΔF1
------------------------------------------------------------
OVERALL                      +0.012     +0.008   +0.010
```

## Data format

Both `train.jsonl` and `test.jsonl` use the privacy-filter span format:

```json
{"text": "Alice Smith's email is alice@example.com ...", "spans": [
  {"start": 0, "end": 11, "label": "private_person"},
  {"start": 22, "end": 42, "label": "private_email"}
]}
```

Labels: `private_person`, `private_email`, `private_phone`, `private_address`,
`private_date`, `account_number`, `private_url`, `secret`.
