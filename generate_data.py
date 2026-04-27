#!/usr/bin/env python3
"""
Generate synthetic PII-annotated data using NVIDIA DataDesigner.

Produces train.jsonl and test.jsonl in the OpenAI privacy-filter span format:
  {"text": "...", "spans": [{"start": 0, "end": 10, "label": "private_person"}]}

Usage:
    python generate_data.py --num-records 500 --output-dir data/
    python generate_data.py --num-records 500 --language en

Requires:
    OPENAI_API_KEY set in environment.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel

import data_designer.config as dd
from data_designer.config.models import ModelConfig
from data_designer.interface import DataDesigner

MODEL = "gpt-5.4-mini"  # cheapest model for testing


# Maps DataDesigner entity fields → privacy-filter label names
LABEL_MAP: dict[str, str] = {
    "name": "private_person",
    "email": "private_email",
    "phone": "private_phone",
    "address": "private_address",
    "date_of_birth": "private_date",
    "account_number": "account_number",
    "personal_url": "private_url",
}

SCENARIOS_SHARED = [
    "medical intake form",
    "job application",
    "customer support ticket",
    "bank account opening form",
    "insurance claim",
    "hotel reservation",
    "legal deposition excerpt",
    "employee onboarding document",
    "online shopping order confirmation",
    "apartment rental application",
    "network security incident report",
    "IT helpdesk ticket",
    "VPN access complaint",
    "online fraud report",
]

SCENARIOS_SV = SCENARIOS_SHARED + [
    "Swedish tax authority form",
    "Swedish healthcare referral",
    "Swedish social services application",
    "Swedish driver's license renewal",
    "Swedish unemployment benefit application",
    "Swedish municipality permit application",
]

SCENARIOS_EN = SCENARIOS_SHARED + [
    "US tax return form",
    "emergency room intake",
    "university enrollment form",
    "vehicle registration",
    "mortgage pre-approval application",
    "passport renewal form",
]

LANG_CONFIG = {
    "sv": {
        "scenarios": SCENARIOS_SV,
        "faker_locale": "sv_SE",
        "pii_prompt": (
            "Generate a realistic set of PII for a Swedish person in a {{ scenario }}. "
            "Return JSON with: name (Swedish full name), email, phone (Swedish format, e.g. 070-123 45 67), "
            "address (Swedish street address with postal code and city, e.g. Storgatan 12, 413 01 Göteborg), "
            "date_of_birth (YYYY-MM-DD), "
            "account_number (8-12 digit number — include only for banking/financial scenarios, otherwise omit), "
            "personal_url (LinkedIn or personal website URL — include for job applications and onboarding, otherwise omit)."
        ),
        "doc_prompt_dd": (
            "Skriv ett realistiskt dokument av typen '{{ scenario }}' på svenska. "
            "Använd EXAKT dessa uppgifter ordagrant (kopiera tecken för tecken): "
            "{{ pii_entities }}. "
            "Dokumentet ska vara 120–280 ord och låta naturligt för sin kontext."
        ),
    },
    "en": {
        "scenarios": SCENARIOS_EN,
        "faker_locale": "en_US",
        "pii_prompt": (
            "Generate a realistic set of PII for a person in a {{ scenario }}. "
            "Return JSON with: name (full name), email, phone (US format with area code), "
            "address (full street address with city/state/zip), date_of_birth (MM/DD/YYYY), "
            "account_number (8-12 digit number — include only for banking/financial scenarios, otherwise omit), "
            "personal_url (LinkedIn or personal website URL — include for job applications and onboarding, otherwise omit)."
        ),
        "doc_prompt_dd": (
            "Write a realistic {{ scenario }} document in English. "
            "Use EXACTLY these details verbatim (copy them character-for-character): "
            "{{ pii_entities }}. "
            "The document should be 120–280 words and read naturally for its context."
        ),
    },
}


class PIIEntities(BaseModel):
    name: str
    email: str
    phone: str
    address: str
    date_of_birth: str
    account_number: Optional[str] = None
    personal_url: Optional[str] = None


def build_faker_seed_df(num_records: int, language: str, seed: int) -> pd.DataFrame:
    from pii_sampler import sample_pii_entities

    cfg = LANG_CONFIG[language]
    random.seed(seed)
    rows = []
    for _ in range(num_records):
        scenario = random.choice(cfg["scenarios"])
        entities = sample_pii_entities(scenario, locale=cfg["faker_locale"])
        rows.append({"scenario": scenario, "pii_entities": entities.model_dump(exclude_none=True)})
    return pd.DataFrame(rows)


def build_config_builder(language: str, seed_df: pd.DataFrame | None = None) -> dd.DataDesignerConfigBuilder:
    cfg = LANG_CONFIG[language]
    model_config = ModelConfig(alias="openai-text", model=MODEL, provider="openai")
    builder = dd.DataDesignerConfigBuilder(model_configs=[model_config])

    if seed_df is not None:
        builder.with_seed_dataset(dd.DataFrameSeedSource(df=seed_df))
    else:
        builder.add_column(
            dd.SamplerColumnConfig(
                name="scenario",
                sampler_type=dd.SamplerType.CATEGORY,
                params=dd.CategorySamplerParams(values=cfg["scenarios"]),
            )
        )
        builder.add_column(
            dd.LLMStructuredColumnConfig(
                name="pii_entities",
                model_alias="openai-text",
                prompt=cfg["pii_prompt"],
                output_format=PIIEntities,
            )
        )

    builder.add_column(
        dd.LLMTextColumnConfig(
            name="document",
            model_alias="openai-text",
            prompt=cfg["doc_prompt_dd"],
        )
    )

    return builder


def find_spans(text: str, entities: dict) -> dict[str, list[list[int]]]:
    """Return spans in opf format: {label: [[start, end], ...]}"""
    spans: dict[str, list[list[int]]] = {}
    for field, label in LABEL_MAP.items():
        value = entities.get(field)
        if not value or not isinstance(value, str) or not value.strip():
            continue
        offsets = [
            [m.start(), m.end()]
            for m in re.finditer(re.escape(value.strip()), text, re.IGNORECASE)
        ]
        if offsets:
            spans[label] = offsets
    return spans


def df_to_jsonl(df) -> list[dict]:
    import numpy as np

    output = []
    for _, row in df.iterrows():
        text = row.get("document", "")
        entities = row.get("pii_entities", {})

        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except json.JSONDecodeError:
                entities = {}
        elif hasattr(entities, "model_dump"):
            entities = entities.model_dump()
        elif isinstance(entities, np.ndarray):
            # DataDesigner returns seed dict columns as array of [key, value] pairs
            entities = dict(entities.tolist())

        if not isinstance(entities, dict):
            entities = {}

        spans = find_spans(str(text), entities)
        if str(text).strip() and spans:
            output.append({"text": str(text), "spans": spans})

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic PII data")
    parser.add_argument("--num-records", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generator",
        choices=["llm", "faker"],
        default="faker",
        help="faker: Faker generates PII (default). llm: DataDesigner generates PII.",
    )
    parser.add_argument(
        "--language",
        choices=["sv", "en"],
        default="sv",
        help="sv: Swedish PII and documents (default). en: English/US PII and documents.",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Error: OPENAI_API_KEY not set.\n  export OPENAI_API_KEY=sk-...")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.generator == "faker":
        print(f"Generating {args.num_records} Faker PII records ({args.language})...")
        seed_df = build_faker_seed_df(args.num_records, args.language, args.seed)
        config_builder = build_config_builder(args.language, seed_df=seed_df)
    else:
        config_builder = build_config_builder(args.language)

    print(f"Generating {args.num_records} documents with DataDesigner ({args.language})...")
    designer = DataDesigner()
    result = designer.create(config_builder=config_builder, num_records=args.num_records)

    df = result.load_dataset()
    jsonl_records = df_to_jsonl(df)
    print(f"Annotated spans on {len(jsonl_records)}/{len(df)} records")

    random.seed(args.seed)
    random.shuffle(jsonl_records)
    split = int(len(jsonl_records) * (1 - args.test_split))
    train_records, test_records = jsonl_records[:split], jsonl_records[split:]

    for name, data in [("train", train_records), ("test", test_records)]:
        path = args.output_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for rec in data:
                f.write(json.dumps(rec) + "\n")
        span_count = sum(len(r["spans"]) for r in data)
        print(f"  {name}.jsonl: {len(data)} records, {span_count} spans → {path}")


if __name__ == "__main__":
    main()
