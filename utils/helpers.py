from __future__ import annotations

import re
from typing import Any


def format_currency(value: str) -> str:
    return value if isinstance(value, str) and value else 'N/A'


def parse_price(value: str) -> int:
    digits = re.sub(r'[^0-9]', '', str(value))
    return int(digits) if digits else 0


def price_range_label(value: str) -> str:
    amount = parse_price(value)
    if amount < 600_000:
        return 'Moins de \u00a3600k'
    if amount < 1_000_000:
        return '\u00a3600k - \u00a3999k'
    if amount < 1_500_000:
        return '\u00a31.0m - \u00a31.49m'
    if amount < 2_000_000:
        return '\u00a31.5m - \u00a31.99m'
    return '\u00a32.0m+'


def similarity_verdict(score: float) -> str:
    if score >= 0.85:
        return 'Très similaire'
    if score >= 0.65:
        return 'Modérément similaire'
    return 'Non relié'


def safe_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0
