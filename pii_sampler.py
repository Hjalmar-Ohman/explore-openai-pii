"""
Generate PIIEntities using official name data and Faker — no LLM involved.

Names are weighted-sampled from SCB (Statistics Sweden) top-name lists
(SCB Namnstatistik 2023). Other fields use Faker with sv_SE locale.
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


# SCB Namnstatistik 2023 — (name, approx. bearer count)
_MALE_NAMES: list[tuple[str, int]] = [
    ("Lars", 86000), ("Karl", 83000), ("Anders", 78000), ("Johan", 72000),
    ("Erik", 71000), ("Per", 68000), ("Nils", 62000), ("Stefan", 58000),
    ("Mikael", 56000), ("Hans", 54000), ("Göran", 51000), ("Jan", 50000),
    ("Björn", 47000), ("Peter", 45000), ("Fredrik", 44000), ("Magnus", 43000),
    ("Sven", 42000), ("Daniel", 38000), ("Mattias", 36000), ("Andreas", 35000),
    ("Marcus", 34000), ("David", 33000), ("Jonas", 32000), ("Martin", 31000),
    ("Patrik", 30000), ("Ulf", 29000), ("Christer", 28000), ("Thomas", 27000),
    ("Kjell", 26000), ("Lennart", 25000),
]

_FEMALE_NAMES: list[tuple[str, int]] = [
    ("Maria", 92000), ("Anna", 88000), ("Kristina", 74000), ("Eva", 70000),
    ("Margareta", 68000), ("Elisabeth", 64000), ("Karin", 62000), ("Ingrid", 59000),
    ("Marie", 55000), ("Monica", 52000), ("Birgitta", 50000), ("Ulla", 48000),
    ("Lena", 46000), ("Sara", 42000), ("Emma", 40000), ("Johanna", 38000),
    ("Sofia", 37000), ("Malin", 36000), ("Linda", 35000), ("Jessica", 34000),
    ("Hanna", 33000), ("Sandra", 32000), ("Lisa", 31000), ("Maja", 30000),
    ("Julia", 29000), ("Ida", 28000), ("Åsa", 27000), ("Camilla", 26000),
    ("Annika", 25000), ("Jenny", 24000),
]

_SURNAMES: list[tuple[str, int]] = [
    ("Johansson", 265000), ("Andersson", 260000), ("Karlsson", 231000),
    ("Nilsson", 175000), ("Eriksson", 151000), ("Larsson", 136000),
    ("Olsson", 108000), ("Persson", 105000), ("Svensson", 104000),
    ("Gustafsson", 97000), ("Pettersson", 96000), ("Jonsson", 78000),
    ("Jansson", 73000), ("Hansson", 68000), ("Bengtsson", 63000),
    ("Jönsson", 59000), ("Lindberg", 52000), ("Jakobsson", 49000),
    ("Magnusson", 47000), ("Olofsson", 46000), ("Lindström", 44000),
    ("Lindqvist", 43000), ("Lindgren", 42000), ("Axelsson", 39000),
    ("Bergström", 37000), ("Lundgren", 36000), ("Fredriksson", 35000),
    ("Sandberg", 34000), ("Mattsson", 33000), ("Gunnarsson", 32000),
]

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


def _weighted_choice(pairs: list[tuple[str, int]]) -> str:
    names, weights = zip(*pairs)
    return random.choices(names, weights=weights, k=1)[0]


def _swedish_name() -> str:
    pool = _MALE_NAMES if random.random() < 0.5 else _FEMALE_NAMES
    return f"{_weighted_choice(pool)} {_weighted_choice(_SURNAMES)}"


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
    """Return a PIIEntities instance sampled from official sources and Faker."""
    fake = _get_faker(locale)

    if locale == "sv_SE":
        name = _swedish_name()
        date_fmt = "%Y-%m-%d"
    else:
        name = fake.name()
        date_fmt = "%m/%d/%Y"

    return PIIEntities(
        name=name,
        email=fake.email(),
        phone=fake.phone_number(),
        address=fake.address().replace("\n", ", "),
        date_of_birth=fake.date_of_birth(minimum_age=18, maximum_age=80).strftime(date_fmt),
        account_number=_account_number() if scenario in _FINANCIAL_SCENARIOS else None,
        personal_url=_linkedin_url(name) if scenario in _URL_SCENARIOS else None,
    )
