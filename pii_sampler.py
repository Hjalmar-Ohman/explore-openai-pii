"""
Generate PIIEntities using Faker — no LLM involved.
"""

from __future__ import annotations

import random
from typing import Optional

from faker import Faker
from pydantic import BaseModel


class PIIEntities(BaseModel):
    name: str
    email: str
    phone: str
    address: str
    date_of_birth: str
    account_number: Optional[str] = None
    personal_url: Optional[str] = None


_FINANCIAL_SCENARIOS = {
    "bank account opening form",
    "insurance claim",
    "online shopping order confirmation",
}

_URL_SCENARIOS = {
    "job application",
    "employee onboarding document",
}

Faker.seed(None)  # use OS entropy so each run is different
_fakers: dict[str, Faker] = {}


def _get_faker(locale: str) -> Faker:
    if locale not in _fakers:
        _fakers[locale] = Faker(locale)
    return _fakers[locale]


def _linkedin_url(name: str) -> str:
    slug = (
        name.lower()
        .replace(" ", "-")
        .replace("å", "a").replace("ä", "a").replace("ö", "o")
        .replace("é", "e").replace("ü", "u")
    )
    return f"https://linkedin.com/in/{slug}-{random.randint(100, 9999)}"


def _account_number() -> str:
    length = random.randint(8, 12)
    return str(random.randint(10 ** (length - 1), 10**length - 1))


def sample_pii_entities(scenario: str, locale: str = "sv_SE") -> PIIEntities:
    fake = _get_faker(locale)
    name = fake.name()
    date_fmt = "%Y-%m-%d" if locale == "sv_SE" else "%m/%d/%Y"

    return PIIEntities(
        name=name,
        email=fake.email(),
        phone=fake.phone_number(),
        address=fake.address().replace("\n", ", "),
        date_of_birth=fake.date_of_birth(minimum_age=18, maximum_age=80).strftime(date_fmt),
        account_number=_account_number() if scenario in _FINANCIAL_SCENARIOS else None,
        personal_url=_linkedin_url(name) if scenario in _URL_SCENARIOS else None,
    )
