from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


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


def _parse_text_for_qty(text: str) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    text = (text or "").strip()
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

    weight_g = max(grams) if grams else None
    volume_ml = max(mls) if mls else None
    return (weight_g, volume_ml, None)


def parse_quantity(
    quantity: Optional[str],
    serving_size: Optional[str],
) -> Tuple[Optional[float], Optional[float], Optional[int], str]:
    """
    Return (weight_g, volume_ml, count, source)

    Pricing MUST be driven by OFF 'quantity' (package size) only.
    Serving size is not package size; treat it as 'none' for pricing/debug.
    """
    if isinstance(quantity, str) and quantity.strip():
        g, ml, count = _parse_text_for_qty(quantity)
        if g is not None or ml is not None or count is not None:
            return (g, ml, count, "quantity")

    # Explicitly ignore serving_size as a package size proxy.
    return (None, None, None, "none")


# ----------------------------
# Bucket + pricing
# ----------------------------

@dataclass(frozen=True)
class Bucket:
    name: str
    unit: str  # "kg" or "l"
    median_unit_price: float  # USD per unit
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
    currency = obj.get("currency", "USD")
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
        buckets["default"] = Bucket(
            name="default", unit="kg", median_unit_price=10.0, sigma=0.55, default_qty_g=250, default_qty_ml=500
        )

    return PricingConfig(
        currency=currency,
        rounding_endings=endings,
        min_price=min_price,
        max_price=max_price,
        premium_multipliers=prem,
        buckets=buckets,
    )


def bucket_from_categories(primary_category: str, categories: Iterable[str], title: Optional[str] = None) -> str:
    """
    Heuristic mapping to pricing buckets.

    IMPORTANT CHANGE:
    - Include title (product_name) in the classifier so items like "Extra virgin olive oil..."
      reliably map to olive_oil even when OFF categories are broad.
    """
    parts = [primary_category, *categories]
    if isinstance(title, str) and title.strip():
        parts.append(title)
    hay = " | ".join(parts).lower()

    if any(k in hay for k in ["snack", "confection", "candy", "chocolate", "sweet", "biscuit", "cookie", "gummies"]):
        return "snacks_sweets"

    if any(k in hay for k in ["condiment", "sauce", "vinegar", "dressing", "marinade", "ketchup", "mustard"]):
        return "condiments_sauces"

    # EVOO / olive oil must be checked BEFORE generic oils_fats
    if "olive oil" in hay or ("extra virgin" in hay and "olive" in hay):
        return "olive_oil"

    if any(k in hay for k in ["oil", "sesame oil", "fat"]):
        return "oils_fats"

    if any(k in hay for k in ["sweetener", "syrup", "maple", "honey"]):
        return "sweeteners_syrups"

    if any(k in hay for k in ["meal", "pasta", "ravioli", "frozen", "ready", "prepared", "dish"]):
        return "meals_chilled_frozen"

    if any(k in hay for k in ["fruit", "vegetable", "produce", "salad", "lettuce", "apple"]):
        return "produce"

    if any(k in hay for k in ["milk", "cheese", "yogurt", "dairy", "butter"]):
        return "dairy"

    if any(k in hay for k in ["beverage", "drink", "juice", "soda", "soft", "water"]):
        return "beverages_soft"

    if any(k in hay for k in ["coffee", "tea"]):
        return "coffee_tea"

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
    mu = math.log(max(median, 1e-9))
    return math.exp(rng.gauss(mu, sigma))


def _apply_retail_rounding(price: float, endings: tuple[float, ...]) -> float:
    if price <= 0:
        return 0.0

    dollars = math.floor(price)
    cents = price - dollars

    endings_sorted = sorted(endings)
    chosen = None
    for e in endings_sorted:
        if e <= cents + 1e-9:
            chosen = e
    if chosen is None:
        dollars = max(0, dollars - 1)
        chosen = endings_sorted[-1]

    return round(dollars + chosen, 2)


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
    title: Optional[str] = None,  # NEW (safe default)
) -> Tuple[float, str, str]:
    
    """
    Returns (price, bucket_name, unit_price_debug_string).
    """
    bucket_name = bucket_from_categories(primary_category, categories, title=title)
    bucket = config.buckets.get(bucket_name, config.buckets["default"])

    weight_g, volume_ml, _count, qty_source = parse_quantity(quantity, serving_size)

    def _default_qty_for_bucket() -> Tuple[float, str]:
        if bucket.unit == "kg":
            if bucket.default_qty_g is not None:
                return (bucket.default_qty_g / 1000.0, f"default {bucket.default_qty_g:.0f}g")
            return (0.25, "fallback 250g")
        if bucket.unit == "l":
            if bucket.default_qty_ml is not None:
                return (bucket.default_qty_ml / 1000.0, f"default {bucket.default_qty_ml:.0f}ml")
            return (0.5, "fallback 500ml")
        return (1.0, "fallback unit")

    if bucket.unit == "kg" and qty_source == "quantity" and weight_g is not None:
        qty_in_unit = weight_g / 1000.0
        qty_debug = f"{weight_g:.0f}g"
    elif bucket.unit == "l" and qty_source == "quantity" and volume_ml is not None:
        qty_in_unit = volume_ml / 1000.0
        qty_debug = f"{volume_ml:.0f}ml"
    else:
        qty_in_unit, qty_debug = _default_qty_for_bucket()
        qty_debug = f"{qty_debug} (no package qty)"

    rng = _seeded_rng(gtin, bucket.name, qty_debug)

    unit_price = _lognormal_with_median(rng, bucket.median_unit_price, bucket.sigma)

    # OPTIONAL: prevent absurdly low EVOO pricing (keeps tails realistic for demos)
    if bucket.name == "olive_oil":
        unit_price = max(unit_price, 0.60 * bucket.median_unit_price)

    # Economy of scale only when quantity is explicit
    scale_debug = ""
    if qty_source == "quantity":
        ref = 0.25 if bucket.unit == "kg" else 0.5
        ratio = max(0.15, min(10.0, qty_in_unit / ref))

        alpha_large = 0.22
        alpha_small = 0.10

        scale = ratio ** (-alpha_large) if ratio >= 1.0 else ratio ** (-alpha_small)
        scale = max(0.55, min(1.35, scale))
        unit_price *= scale
        scale_debug = f", scale={scale:.2f}, ratio={ratio:.2f}"

    mult = 1.0
    if labels_tags:
        for t in labels_tags:
            if t in config.premium_multipliers:
                mult *= config.premium_multipliers[t]

    mult *= 1.05 if (brand and brand.strip()) else 0.97
    unit_price *= mult

    price = unit_price * qty_in_unit
    price = max(config.min_price, min(config.max_price, price))
    price = _apply_retail_rounding(price, config.rounding_endings)

    unit_debug = (
        f"{unit_price:.2f} {config.currency}/{bucket.unit} "
        f"({qty_debug}, source={qty_source}, bucket={bucket.name}{scale_debug})"
    )
    return price, bucket.name, unit_debug