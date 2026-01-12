from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


# ----------------------------
# Quantity parsing
# ----------------------------

_QTY_RE = re.compile(
    r"(?P<num>\d+(?:[\.,]\d+)?)\s*(?P<unit>kg|g|mg|lb|lbs|oz|l|ml|cl|dl|fl\s*oz|floz)\b",
    re.IGNORECASE,
)

_MULTI_RE = re.compile(
    r"(?P<count>\d+)\s*[x×]\s*(?P<num>\d+(?:[\.,]\d+)?)\s*(?P<unit>kg|g|mg|oz|lb|lbs|l|ml|cl|dl|fl\s*oz|floz)\b",
    re.IGNORECASE,
)


def _to_float(s: str) -> float:
    return float(s.replace(",", ".").strip())


def _unit_to_g(num: float, unit: str) -> Optional[float]:
    u = unit.lower().replace(" ", "")
    if u == "kg":
        return num * 1000.0
    if u == "g":
        return num
    if u == "mg":
        return num / 1000.0
    if u in ("lb", "lbs"):
        return num * 453.59237
    if u == "oz":
        return num * 28.349523125
    return None


def _unit_to_ml(num: float, unit: str) -> Optional[float]:
    u = unit.lower().replace(" ", "")
    if u == "l":
        return num * 1000.0
    if u == "ml":
        return num
    if u == "cl":
        return num * 10.0
    if u == "dl":
        return num * 100.0
    if u in ("floz", "fl oz"):
        return num * 29.5735295625
    return None


def parse_quantity(quantity: Optional[str], serving_size: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Return (weight_g, volume_ml, count) when parseable.

    Supports:
      - "3 oz", "946 ml", "0.75 oz (22 g)"
      - "6 x 330 ml", "12x250ml"
    """
    candidates = []
    for s in (quantity, serving_size):
        if isinstance(s, str) and s.strip():
            candidates.append(s)

    text = " | ".join(candidates)
    if not text:
        return (None, None, None)

    # Multipacks like "6 x 330 ml"
    m = _MULTI_RE.search(text)
    if m:
        count = int(m.group("count"))
        num = _to_float(m.group("num"))
        unit = m.group("unit")
        g = _unit_to_g(num, unit)
        ml = _unit_to_ml(num, unit)
        if g is not None:
            return (g * count, None, count)
        if ml is not None:
            return (None, ml * count, count)

    # If text contains a parenthetical with g/ml, prefer that (often more precise)
    # We'll just scan all unit expressions and pick the "best" based on presence.
    grams = []
    mls = []
    for mm in _QTY_RE.finditer(text):
        num = _to_float(mm.group("num"))
        unit = mm.group("unit")
        g = _unit_to_g(num, unit)
        ml = _unit_to_ml(num, unit)
        if g is not None:
            grams.append(g)
        if ml is not None:
            mls.append(ml)

    # Heuristic: if both appear, prefer grams (foods) unless grams is tiny and ml is plausible
    weight_g = max(grams) if grams else None
    volume_ml = max(mls) if mls else None

    return (weight_g, volume_ml, None)


# ----------------------------
# Bucket + pricing
# ----------------------------

@dataclass(frozen=True)
class Bucket:
    name: str
    unit: str  # "kg" or "l"
    median_unit_price: float  # EUR per unit
    sigma: float  # lognormal sigma
    default_qty_g: Optional[float] = None
    default_qty_ml: Optional[float] = None


@dataclass(frozen=True)
class PricingConfig:
    currency: str
    rounding_endings: tuple[float, ...]
    min_price: float
    max_price: float
    premium_multipliers: dict[str, float]
    buckets: dict[str, Bucket]


def load_pricing_config(path: Path) -> PricingConfig:
    obj = json.loads(path.read_text(encoding="utf-8"))
    currency = obj.get("currency", "EUR")
    endings = tuple(float(x) for x in obj.get("rounding_endings", [0.99, 0.49, 0.29]))
    min_price = float(obj.get("min_price", 0.49))
    max_price = float(obj.get("max_price", 99.99))

    prem = {str(k): float(v) for k, v in (obj.get("premium_multipliers") or {}).items()}

    buckets_raw = obj.get("buckets") or {}
    buckets: dict[str, Bucket] = {}
    for name, b in buckets_raw.items():
        buckets[name] = Bucket(
            name=name,
            unit=str(b.get("unit", "kg")),
            median_unit_price=float(b.get("median_unit_price", 10.0)),
            sigma=float(b.get("sigma", 0.55)),
            default_qty_g=float(b.get("default_qty_g")) if b.get("default_qty_g") is not None else None,
            default_qty_ml=float(b.get("default_qty_ml")) if b.get("default_qty_ml") is not None else None,
        )

    if "default" not in buckets:
        buckets["default"] = Bucket(name="default", unit="kg", median_unit_price=10.0, sigma=0.55, default_qty_g=250)

    return PricingConfig(
        currency=currency,
        rounding_endings=endings,
        min_price=min_price,
        max_price=max_price,
        premium_multipliers=prem,
        buckets=buckets,
    )


def bucket_from_categories(primary_category: str, categories: Iterable[str]) -> str:
    """
    Heuristic mapping from your humanized OFF categories to pricing buckets.
    Tune as needed.
    """
    hay = " | ".join([primary_category, *categories]).lower()

    # Snacks / sweets
    if any(k in hay for k in ["snack", "confection", "candy", "chocolate", "sweet", "biscuit", "cookie", "gummies"]):
        return "snacks_sweets"

    # Condiments / sauces / vinegars / dressings
    if any(k in hay for k in ["condiment", "sauce", "vinegar", "dressing", "marinade", "ketchup", "mustard"]):
        return "condiments_sauces"

    # Oils / fats
    if any(k in hay for k in ["oil", "olive oil", "sesame oil", "fat"]):
        return "oils_fats"

    # Sweeteners / syrups
    if any(k in hay for k in ["sweetener", "syrup", "maple", "honey"]):
        return "sweeteners_syrups"

    # Meals / prepared / frozen
    if any(k in hay for k in ["meal", "pasta", "ravioli", "frozen", "ready", "prepared", "dish"]):
        return "meals_chilled_frozen"

    # Produce
    if any(k in hay for k in ["fruit", "vegetable", "produce", "salad", "lettuce", "apple"]):
        return "produce"

    # Dairy
    if any(k in hay for k in ["milk", "cheese", "yogurt", "dairy", "butter"]):
        return "dairy"

    # Beverages (non-alcohol)
    if any(k in hay for k in ["beverage", "drink", "juice", "soda", "soft", "water"]):
        return "beverages_soft"

    # Coffee / tea
    if any(k in hay for k in ["coffee", "tea"]):
        return "coffee_tea"

    # Bakery
    if any(k in hay for k in ["bread", "bakery", "croissant", "cake", "pastry"]):
        return "bakery"

    return "default"


def _seeded_rng(*parts: str) -> random.Random:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    seed = int.from_bytes(h.digest(), "big")
    return random.Random(seed)


def _lognormal_with_median(rng: random.Random, median: float, sigma: float) -> float:
    # lognormal: exp(N(mu, sigma)), median = exp(mu) => mu = ln(median)
    mu = math.log(max(median, 1e-9))
    return math.exp(rng.gauss(mu, sigma))


def _apply_retail_rounding(price: float, endings: tuple[float, ...]) -> float:
    """
    Round down to nearest retail-ish ending in the same euro.
    Example: 2.83 -> 2.79? (if endings include .79)
    We use endings like .99, .49, .29 etc.
    """
    if price <= 0:
        return 0.0

    euros = math.floor(price)
    cents = price - euros

    # pick the largest ending <= cents; otherwise go down 1 euro and use max ending
    endings_sorted = sorted(endings)
    chosen = None
    for e in endings_sorted:
        if e <= cents + 1e-9:
            chosen = e
    if chosen is None:
        euros = max(0, euros - 1)
        chosen = endings_sorted[-1]

    return round(euros + chosen, 2)


def estimate_price(
    *,
    gtin: str,
    primary_category: str,
    categories: Iterable[str],
    quantity: Optional[str],
    serving_size: Optional[str],
    labels_tags: Optional[list[str]],
    brand: Optional[str],
    config: PricingConfig,
) -> Tuple[float, str, str]:
    """
    Returns (price, bucket_name, unit_price_debug_string).

    Price is deterministic for a given (gtin, bucket, quantity-ish).
    """
    bucket_name = bucket_from_categories(primary_category, categories)
    bucket = config.buckets.get(bucket_name, config.buckets["default"])

    # Parse sizes
    weight_g, volume_ml, _count = parse_quantity(quantity, serving_size)

    # Determine quantity in bucket unit
    qty_in_unit: Optional[float] = None
    qty_debug = ""

    if bucket.unit == "kg":
        if weight_g is not None:
            qty_in_unit = weight_g / 1000.0
            qty_debug = f"{weight_g:.0f}g"
        elif bucket.default_qty_g is not None:
            qty_in_unit = bucket.default_qty_g / 1000.0
            qty_debug = f"default {bucket.default_qty_g:.0f}g"
        else:
            qty_in_unit = 0.25
            qty_debug = "fallback 250g"
    elif bucket.unit == "l":
        if volume_ml is not None:
            qty_in_unit = volume_ml / 1000.0
            qty_debug = f"{volume_ml:.0f}ml"
        elif bucket.default_qty_ml is not None:
            qty_in_unit = bucket.default_qty_ml / 1000.0
            qty_debug = f"default {bucket.default_qty_ml:.0f}ml"
        else:
            qty_in_unit = 0.5
            qty_debug = "fallback 500ml"
    else:
        # Shouldn't happen, but keep safe
        qty_in_unit = 1.0
        qty_debug = "fallback unit"

    # Deterministic RNG: tie to gtin + bucket + qty_debug
    rng = _seeded_rng(gtin, bucket.name, qty_debug)

    # Sample unit price (EUR per kg or per L)
    unit_price = _lognormal_with_median(rng, bucket.median_unit_price, bucket.sigma)

    # Mild economy of scale: larger packs slightly cheaper per unit (cap effect)
    # (If we used a default qty, keep it neutral.)
    if "default" not in qty_debug:
        ref = 0.25 if bucket.unit == "kg" else 0.5
        ratio = max(0.25, min(4.0, qty_in_unit / ref))
        # larger ratio => cheaper per unit; smaller => slightly pricier
        unit_price *= ratio ** (-0.10)

    # Apply label premiums
    mult = 1.0
    if labels_tags:
        for t in labels_tags:
            if t in config.premium_multipliers:
                mult *= config.premium_multipliers[t]

    # Brand presence: slight premium for branded items
    if brand and brand.strip():
        mult *= 1.05
    else:
        mult *= 0.97

    unit_price *= mult

    # Pack price
    price = unit_price * qty_in_unit

    # Clamp and round
    price = max(config.min_price, min(config.max_price, price))
    price = _apply_retail_rounding(price, config.rounding_endings)

    unit_debug = f"{unit_price:.2f} {config.currency}/{bucket.unit} ({qty_debug}, bucket={bucket.name})"
    return price, bucket.name, unit_debug