from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")
_SPACE_RE = re.compile(r"\s+")

STOPWORDS = {
    "and", "or", "the", "a", "an", "with", "without", "from", "of", "to", "in", "on",
    "for", "by", "contains", "contain", "may", "include", "ingredients", "ingredient",
    "natural", "artificial", "flavour", "flavouring", "flavors", "flavoring", "added",
    "water", "salt", "sugar", "acid", "spices", "spice", "extract", "powder"
}

# Ingredient -> descriptor buckets (keep generic; avoid health/quality claims)
INGREDIENT_TO_TAG = {
    "cocoa": "chocolate",
    "chocolate": "chocolate",
    "hazelnut": "hazelnut",
    "almond": "almond",
    "peanut": "peanut",
    "vanilla": "vanilla",
    "strawberry": "strawberry",
    "raspberry": "raspberry",
    "blueberry": "blueberry",
    "lemon": "citrus",
    "lime": "citrus",
    "orange": "citrus",
    "mint": "mint",
    "cinnamon": "spiced",
    "ginger": "spiced",
    "garlic": "garlic",
    "tomato": "tomato",
    "vinegar": "tangy",
    "pepper": "spicy",
    "chili": "spicy",
    "chilli": "spicy",
    "sesame": "sesame",
    "olive": "olive",
    "coffee": "coffee",
    "tea": "tea",
    "milk": "dairy",
    "cheese": "dairy",
    "yogurt": "dairy",
    "butter": "dairy",
    "beef": "meat",
    "chicken": "meat",
    "pork": "meat",
    "fish": "seafood",
    "shrimp": "seafood",
    "crab": "seafood",
    "lobster": "seafood",
}

# Category bucket mapping using your observed top categories
CATEGORY_BUCKETS = {
    "snacks": "snacks",
    "sweet snacks": "snacks",
    "confectioneries": "snacks",
    "cocoa and its products": "snacks",
    "biscuits and cakes": "bakery",
    "breakfasts": "breakfast",
    "beverages": "beverages",
    "beverages and beverages preparations": "beverages",
    "condiments": "condiments",
    "sauces": "condiments",
    "meals": "meals",
    "frozen foods": "meals",
    "meats and their products": "meat",
    "dairies": "dairy",
    "fermented milk products": "dairy",
    "fermented foods": "dairy",
    "fruits and vegetables based foods": "produce",
    "cereals and potatoes": "pantry",
    "plant based foods and beverages": "pantry",
    "plant based foods": "pantry",
}

DEFAULT_PHRASES: Dict[str, list[str]] = {
    "snacks": [
        "A snack option for sharing or on-the-go.",
        "Works as a quick treat or snack.",
        "Suitable for lunchboxes and small breaks.",
    ],
    "bakery": [
        "A baked option for snacks and desserts.",
        "Pairs well with coffee or tea.",
        "Works for breakfast or a quick snack.",
    ],
    "breakfast": [
        "A breakfast-friendly option for mornings.",
        "Works well with milk, yogurt, or fruit.",
        "Suitable for quick breakfasts.",
    ],
    "beverages": [
        "A drink option for everyday use.",
        "Best served chilled or at room temperature.",
        "Works for meals or snacks.",
    ],
    "condiments": [
        "A pantry staple for everyday cooking.",
        "Works well with salads, sandwiches, and meals.",
        "Suitable for seasoning and finishing dishes.",
    ],
    "meals": [
        "A meal option for quick preparation.",
        "Suitable for lunch or dinner.",
        "Works for convenient meals.",
    ],
    "meat": [
        "A protein option for meals and recipes.",
        "Suitable for cooking and meal prep.",
        "Works in a variety of dishes.",
    ],
    "dairy": [
        "A dairy option for snacks or meals.",
        "Works well in breakfast and desserts.",
        "Suitable for everyday use.",
    ],
    "produce": [
        "A fruit/vegetable option for meals and snacks.",
        "Works well as a side or ingredient.",
        "Suitable for everyday use.",
    ],
    "pantry": [
        "A pantry staple for everyday use.",
        "Works well in cooking and recipes.",
        "Suitable for home cooking.",
    ],
    "default": [
        "A product option for everyday use.",
        "Suitable for meals and snacks.",
        "Works well in a variety of contexts.",
    ],
}

@dataclass(frozen=True)
class DescribeConfig:
    # If provided and exists, overrides DEFAULT_PHRASES partially/fully.
    phrase_bank_path: Optional[Path] = None
    max_phrases: int = 2
    max_tags: int = 4


def _seeded_rng(seed_text: str) -> random.Random:
    h = hashlib.blake2b(seed_text.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(h, "big")
    return random.Random(seed)


def _normalize_space(s: str) -> str:
    return _SPACE_RE.sub(" ", s).strip()


def _bucket_from_categories(categories: Iterable[str]) -> str:
    for c in categories:
        if not isinstance(c, str):
            continue
        key = c.strip().lower()
        if key in CATEGORY_BUCKETS:
            return CATEGORY_BUCKETS[key]
    return "default"


def _extract_terms(text: str, limit: int = 8) -> list[str]:
    words = []
    for m in _WORD_RE.finditer(text.lower()):
        w = m.group(0).strip("-'")
        if len(w) < 3 or w in STOPWORDS:
            continue
        words.append(w)
    # frequency
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    # sort by freq desc, then alpha
    top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in top[:limit]]


def _terms_to_tags(terms: Iterable[str], max_tags: int) -> list[str]:
    tags = []
    seen = set()
    for t in terms:
        tag = INGREDIENT_TO_TAG.get(t)
        if not tag:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
        if len(tags) >= max_tags:
            break
    return tags


def _load_phrase_bank(path: Path) -> Dict[str, list[str]]:
    """
    Expected JSON shape:
    {
      "snacks": ["...", "..."],
      "beverages": ["...", "..."],
      ...
    }
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, list[str]] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) for x in v):
                out[k] = [x.strip() for x in v if x.strip()]
    return out


def generate_description(
    *,
    gtin: str,
    title: str,
    brand: str,
    categories: list[str],
    dietary_restrictions: list[str],
    ingredients_text: Optional[str],
    attrs: Dict[str, Any],
    cfg: DescribeConfig,
) -> str:
    """
    Returns a short, readable description intended for demos.
    Falls back to ingredients/attrs only if needed.
    """
    rng = _seeded_rng(gtin)

    bucket = _bucket_from_categories(categories)

    phrase_bank = DEFAULT_PHRASES
    if cfg.phrase_bank_path and cfg.phrase_bank_path.exists():
        # overlay external phrase bank
        ext = _load_phrase_bank(cfg.phrase_bank_path)
        phrase_bank = {**DEFAULT_PHRASES, **ext}

    # Sentence 1: title + brand (if present)
    brand_part = f" by {brand.strip()}" if brand.strip() else ""
    s1 = f"{title.strip()}{brand_part}."

    # Sentence 2: category-based usage sentence (deterministic pick)
    candidates = phrase_bank.get(bucket) or phrase_bank["default"]
    s2 = rng.choice(candidates)

    # Optional: tags from ingredients/title for flavor/profile line
    tags: list[str] = []
    title_terms = _extract_terms(title, limit=6)
    tags.extend(_terms_to_tags(title_terms, cfg.max_tags))

    if ingredients_text:
        ing_terms = _extract_terms(ingredients_text, limit=12)
        tags.extend(_terms_to_tags(ing_terms, cfg.max_tags))

    # de-dupe tags preserving order
    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]

    extra_sentences: list[str] = []

    if tags:
        extra_sentences.append("Notes: " + ", ".join(tags[: cfg.max_tags]) + ".")

    # Dietary: keep factual language
    if dietary_restrictions:
        extra_sentences.append("Dietary: " + ", ".join(dietary_restrictions) + ".")

    # Quantity/serving if available (short)
    qty = attrs.get("Quantity") or ""
    if isinstance(qty, str) and qty.strip() and len(qty.strip()) <= 32:
        extra_sentences.append(f"Pack size: {qty.strip()}.")

    # Choose up to cfg.max_phrases extra sentences deterministically
    rng.shuffle(extra_sentences)
    extra_sentences = extra_sentences[: cfg.max_phrases]

    # Assemble
    parts = [s1, s2] + extra_sentences
    out = _normalize_space(" ".join(parts))

    # Fallback: if we somehow produced something too short, use a trimmed ingredient snippet
    if len(out) < 25:
        if ingredients_text and ingredients_text.strip():
            snippet = _normalize_space(ingredients_text.strip())
            snippet = snippet[:160].rstrip(",;:. ")
            out = _normalize_space(f"{s1} Ingredients: {snippet}.")
        else:
            out = s1

    return out


def generate_search_text(
    *,
    title: str,
    brand: str,
    categories: list[str],
    dietary_restrictions: list[str],
    ingredients_text: Optional[str],
    attrs: Dict[str, Any],
    cfg: DescribeConfig,
) -> str:
    """
    Denormalized field intended to improve matching.
    """
    terms: list[str] = []
    if title:
        terms.append(title)
    if brand:
        terms.append(brand)
    if categories:
        terms.extend(categories[:3])
    if dietary_restrictions:
        terms.extend(dietary_restrictions)

    if ingredients_text:
        ing_terms = _extract_terms(ingredients_text, limit=10)
        terms.extend(ing_terms)

    # Add a few safe attribute values that users might search for
    for k in ("Nutri-Score", "Eco-Score", "NOVA group", "Allergens", "Labels"):
        v = attrs.get(k)
        if isinstance(v, str) and v.strip():
            terms.append(v.strip())

    return _normalize_space(" ".join(terms))