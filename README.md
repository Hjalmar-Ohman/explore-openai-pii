# explore-openai-pii

Evaluate OpenAI's `openai/privacy-filter` model on synthetic PII data. Generate annotated documents with NVIDIA DataDesigner + Faker, run inference locally or on Modal GPU, and compute span-level F1.

## Pipeline

```
generate_data.py  →  predict.py  →  evaluate.py
       ↓                  ↓               ↓
 data/train.jsonl   predictions.jsonl   P/R/F1 table
 data/test.jsonl
```

Fine-tuning (optional) sits between generate and predict:

```
opf train data/train.jsonl --output finetuned_model
python predict.py --model finetuned_model --output data/predictions_finetuned.jsonl
python evaluate.py --compare data/predictions.jsonl data/predictions_finetuned.jsonl
```

## Setup

**1. Install dependencies:**

```bash
pip install -r requirements.txt
```

**2. Set API keys:**

```bash
export OPENAI_API_KEY=sk-...
```

## Step by step

**Step 1 — generate synthetic data**

```bash
# Swedish (default), Faker PII, 500 records
python generate_data.py --num-records 500 --output-dir data/faker/swedish

# English, Faker PII
python generate_data.py --num-records 500 --language en --output-dir data/faker/english

# Let the LLM generate PII instead of Faker
python generate_data.py --num-records 500 --generator llm --output-dir data/llm/swedish
```

Flags:
- `--language sv|en` — Swedish (default) or English/US PII and documents
- `--generator faker|llm` — Faker generates PII fields (default, cheaper); `llm` lets DataDesigner do it
- `--num-records N` — total records before train/test split
- `--test-split FLOAT` — fraction held out for test (default 0.2)
- `--output-dir PATH` — where to write `train.jsonl` and `test.jsonl`

**Step 2 — run inference**

Locally (MPS/CPU/CUDA auto-detected):

```bash
OPF_MOE_TRITON=0 python predict.py --test-data data/faker/swedish/test.jsonl
OPF_MOE_TRITON=0 python predict.py --model finetuned_model --output data/predictions_finetuned.jsonl
```

On Modal (T4 GPU):

```bash
modal run modal_predict.py --test-data data/faker/swedish/test.jsonl
```

**Step 3 — evaluate**

```bash
# Single model
python evaluate.py --predictions data/predictions.jsonl

# Compare base vs fine-tuned
python evaluate.py --predictions data/predictions.jsonl \
                   --compare data/predictions_finetuned.jsonl

# Show per-label errors (FP/FN examples)
python evaluate.py --predictions data/predictions.jsonl --show-errors private_phone
```

**Interactive demo**

```bash
OPF_MOE_TRITON=0 python demo.py
```

Highlights detected PII inline, shows confidence scores, and flags near-miss tokens (tokens close to the PII threshold).

## Data format

`train.jsonl` / `test.jsonl` use the OPF span format — `spans` is a dict mapping label → list of `[start, end]` character offsets:

```json
{"text": "Alice Smith's email is alice@example.com ...", "spans": {
  "private_person": [[0, 11]],
  "private_email":  [[22, 42]]
}}
```

`predictions.jsonl` adds `gold_spans` and `predicted_spans` side by side:

```json
{"text": "...", "gold_spans": {...}, "predicted_spans": {...}}
```

## Labels

`private_person`, `private_email`, `private_phone`, `private_address`,
`private_date`, `account_number`, `private_url`
