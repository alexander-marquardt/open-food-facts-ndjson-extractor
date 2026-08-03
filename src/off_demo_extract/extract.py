from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Set, TextIO, Tuple

from off_demo_extract.category_tags import (
    CategoryVocabulary,
    TagCurationAudit,
    curate_category_tags,
)
from off_demo_extract.pricing import load_pricing_config, estimate_price
from off_demo_extract.taxonomy import (
    ensure_taxonomy,
    load_taxonomy,
    build_canonical_parent_map,
    category_path_entries,
    default_keep_prefixes,
    display_label,
    global_roots,
    unanchored_head,
    AddressAudit,
    RootAnchorAudit,
)


IMAGE_BASE = "https://images.openfoodfacts.org/images/products"

# Values we should treat as "not meaningful" and avoid emitting in attrs/description.
_UNDEFINED_LIKE = {"undefined", "unknown", "null", "none", "n/a", "na", ""}
MAX_NUM_CATEGORIES = 20


# ----------------------------
# Repo / IO helpers
# ----------------------------

def repo_root() -> Path:
    """
    Find repo root by walking up from this file and looking for *project-root markers*.
    We require both:
      - pyproject.toml
      - src/ directory
    This prevents accidentally treating ./data (or other subdirs) as the repo root.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def open_maybe_gzip(path: Path, encoding: str = "utf-8") -> TextIO:
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding=encoding, errors="replace")
    return path.open("r", encoding=encoding, errors="replace")


# ----------------------------
# Text normalization helpers
# ----------------------------

def _is_undefined_like(value: Optional[str]) -> bool:
    if value is None:
        return True
    v = value.strip().lower()
    return v in _UNDEFINED_LIKE


def _is_mostly_uppercase(s: str, threshold: float = 0.80) -> bool:
    """
    Heuristic: return True when the text is overwhelmingly uppercase letters.
    We only use this to "de-shout" ingredients/description/title that come through
    in all-caps from OFF.
    """
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return (upper / len(letters)) >= threshold


def _deshout_text(s: str) -> str:
    """
    Convert shouty ALL CAPS text to a more natural sentence case.
    Conservative: only applied when _is_mostly_uppercase() is True.
    """
    if not s:
        return s
    if not _is_mostly_uppercase(s):
        return s.strip()

    lowered = s.strip().lower()
    # Sentence-case: capitalize only the first character if present.
    return lowered[:1].upper() + lowered[1:]


# ----------------------------
# OFF image URL construction
# ----------------------------

def pad_gtin13(code: str) -> str:
    s = "".join(ch for ch in code if ch.isdigit())
    if len(s) >= 13:
        return s[:13]
    return ("0" * (13 - len(s))) + s


def product_folder_from_code(code: str) -> str:
    c = pad_gtin13(code)
    return f"{c[0:3]}/{c[3:6]}/{c[6:9]}/{c[9:]}"


def pick_image_resolution_from_sizes(sizes: Dict[str, Any]) -> str:
    if not isinstance(sizes, dict):
        return "full"
    for res in ("400", "200", "100"):
        if res in sizes:
            return res
    return "full"


def build_selected_image_url(code: str, key: str, rev: str, sizes: Dict[str, Any]) -> str:
    res = pick_image_resolution_from_sizes(sizes)
    folder = product_folder_from_code(code)
    return f"{IMAGE_BASE}/{folder}/{key}.{rev}.{res}.jpg"


def build_raw_image_url(code: str, imgid: str, sizes: Dict[str, Any]) -> str:
    res = pick_image_resolution_from_sizes(sizes)
    folder = product_folder_from_code(code)
    if res == "full":
        return f"{IMAGE_BASE}/{folder}/{imgid}.jpg"
    return f"{IMAGE_BASE}/{folder}/{imgid}.{res}.jpg"


def choose_front_key(
    images: Dict[str, Any],
    prefer_lang: str = "en",
    require_lang: Optional[str] = None,
) -> Optional[str]:
    if not isinstance(images, dict) or not images:
        return None

    if require_lang:
        k = f"front_{require_lang}"
        return k if k in images else None

    preferred = f"front_{prefer_lang}"
    if preferred in images:
        return preferred

    for k in images.keys():
        if k.startswith("front_") and len(k) == len("front_") + 2:
            return k

    if "front" in images:
        return "front"

    return None


def compute_image_url(
    product: Dict[str, Any],
    prefer_lang: str = "en",
    require_front_lang: Optional[str] = None,
) -> Optional[str]:
    code = str(product.get("code") or product.get("_id") or "").strip()
    if not code:
        return None

    images = product.get("images")
    if not isinstance(images, dict) or not images:
        return None

    front_key = choose_front_key(images, prefer_lang=prefer_lang, require_lang=require_front_lang)
    if not front_key:
        return None

    sel = images.get(front_key)
    if not isinstance(sel, dict):
        return None

    rev = sel.get("rev")
    sel_sizes = sel.get("sizes") if isinstance(sel.get("sizes"), dict) else {}

    if rev is not None:
        return build_selected_image_url(code=code, key=front_key, rev=str(rev), sizes=sel_sizes)

    imgid = sel.get("imgid")
    if imgid is None:
        return None

    raw = images.get(str(imgid))
    raw_sizes = raw.get("sizes") if isinstance(raw, dict) and isinstance(raw.get("sizes"), dict) else {}
    return build_raw_image_url(code=code, imgid=str(imgid), sizes=raw_sizes)


# ----------------------------
# Language-filtered fields
# ----------------------------

def get_title(product: Dict[str, Any], lang: str = "en") -> Optional[str]:
    t = product.get(f"product_name_{lang}")
    if isinstance(t, str) and t.strip():
        return _deshout_text(t)

    prod_lang = product.get("lang") or product.get("lc")
    if prod_lang == lang:
        t2 = product.get("product_name")
        if isinstance(t2, str) and t2.strip():
            return _deshout_text(t2)

    return None


def get_description(product: Dict[str, Any], lang: str = "en", max_len: int = 600) -> Optional[str]:
    for k in (f"generic_name_{lang}", f"ingredients_text_{lang}"):
        v = product.get(k)
        if isinstance(v, str) and v.strip():
            return _deshout_text(v.strip()[:max_len])

    prod_lang = product.get("lang") or product.get("lc")
    if prod_lang == lang:
        for k in ("generic_name", "ingredients_text"):
            v = product.get(k)
            if isinstance(v, str) and v.strip():
                return _deshout_text(v.strip()[:max_len])

    return None


# ----------------------------
# Categories
# ----------------------------

def extract_categories_tags(product: Dict[str, Any]) -> list[str]:
    cats = product.get("categories_tags")
    if isinstance(cats, list):
        out: list[str] = []
        for x in cats:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def pick_primary_category_tag(tags: list[str]) -> Optional[str]:
    """The product's primary category: the first tag the run accepted.

    ``tags`` must already have been through
    :func:`off_demo_extract.category_tags.curate_category_tags`, which is what
    keeps a refused tag out of ``attrs["Category"]`` and out of the generated
    description — both of which are indexed and searchable, so validating only
    the ``categories`` list would leave the same junk searchable by another route.
    """
    return tags[0] if tags else None


def build_category_label_entries(
    primary_tag: Optional[str],
    tags_filtered: list[str],
    taxonomy: Optional[Dict[str, Any]],
    lang: str = "en",
    max_n: int = MAX_NUM_CATEGORIES,
    vocabulary: Optional[CategoryVocabulary] = None,
) -> list[tuple[str, str]]:
    """``(tag_id, label)`` for the flat ``categories`` field, primary tag first.

    The label comes from :func:`off_demo_extract.taxonomy.display_label` — the
    same function that names a ``category_path`` segment. This field used to
    de-slug the tag id itself, which rendered the identical node under a
    different string in the two fields (``Plant based foods`` here versus
    ``Plant-based foods`` there) and made them unjoinable by string; deriving
    both from one function is what stops a change to labelling from reaching one
    field and not the other.

    Ids ride along so a run can audit the two fields against each other. Callers
    that only want the strings want :func:`build_categories_list`.

    ``vocabulary`` is the set of ids this run may emit. Every tag is checked
    against it before it is labelled, because this function is what writes the
    searchable field and an unvalidated tag reaching it is precisely the defect:
    ``Groceries`` was searchable on 6,299 documents of the built English catalog
    with no node in the taxonomy, no path, and no possible policy value. Callers
    in this module hand over already-curated tags, so the check is normally a
    no-op — it is the last line of defence, not the first. Note that a tag with
    no taxonomy node has no ``name`` for :func:`display_label` to render either,
    so there is nothing to label it *with* that would not be a second, divergent
    labelling rule — the thing #17 removed.

    ``vocabulary`` of ``None`` means no taxonomy was loaded, so there is nothing
    to validate against and the labels are emitted as before.
    """
    tax = taxonomy if taxonomy is not None else {}
    seen: Set[str] = set()
    out: list[tuple[str, str]] = []

    def add(tag: str) -> None:
        if vocabulary is not None and tag not in vocabulary.eligible:
            return
        label = display_label(tax, tag, lang)
        # Suppress "Undefined"/"Unknown"/etc from appearing as categories.
        if _is_undefined_like(label):
            return
        if label not in seen:
            seen.add(label)
            out.append((tag, label))

    if primary_tag:
        add(primary_tag)

    for t in tags_filtered:
        if len(out) >= max_n:
            break
        add(t)

    return out


def build_categories_list(
    primary_tag: Optional[str],
    tags_filtered: list[str],
    taxonomy: Optional[Dict[str, Any]] = None,
    lang: str = "en",
    max_n: int = MAX_NUM_CATEGORIES,
    vocabulary: Optional[CategoryVocabulary] = None,
) -> list[str]:
    return [
        label
        for _tag, label in build_category_label_entries(
            primary_tag, tags_filtered, taxonomy, lang, max_n, vocabulary
        )
    ]


# ----------------------------
# Dietary restrictions (efficient keyword list)
# ----------------------------

def dietary_restrictions_from_off(product: Dict[str, Any]) -> list[str]:
    """
    Return a list of dietary restriction keyword tags suitable for efficient filtering.
    Positive-only (no maybe/unknown flags).
    """
    labels = set(product.get("labels_tags") or [])
    analysis = set(product.get("ingredients_analysis_tags") or [])

    tags: set[str] = set()

    if "en:vegan" in labels or "en:vegan" in analysis:
        tags.add("vegan")
    if "en:vegetarian" in labels or "en:vegetarian" in analysis:
        tags.add("vegetarian")

    if "en:halal" in labels:
        tags.add("halal")
    if "en:kosher" in labels:
        tags.add("kosher")

    if "en:gluten-free" in labels:
        tags.add("gluten_free")
    if "en:lactose-free" in labels:
        tags.add("lactose_free")

    if "en:organic" in labels or "en:usda-organic" in labels:
        tags.add("organic")

    return sorted(tags)


# ----------------------------
# Margin & popularity (for function_score boosting)
# ----------------------------
#
# Both fields feed Elasticsearch ``function_score`` / ``field_value_factor``
# arms with an ``ln1p`` modifier. They used to be a uniform random draw seeded
# on the GTIN, which exercised that code path but demonstrated nothing: moving
# the popularity weight reordered the results and the new order could not be
# explained to anyone watching, because the number underneath was noise
# (elastic/prism#5027).
#
# They are now derived, and they are NOT equally honest — which is why each
# product records where its number came from:
#
#   popularity  OBSERVED   ``unique_scans_n`` from this very dump: how many
#                          distinct people scanned the barcode with the Open
#                          Food Facts app. Real per-product behavioural data.
#   margin      MODELLED   a per-category rate lifted by the product's real
#                          premium / free-from label tags. Open Food Facts
#                          carries no cost data, so nothing can make this one
#                          observed; it is labelled rather than dressed up.
#
# Deriving margin from the estimated unit price was rejected: that price is
# itself a seeded lognormal draw around the bucket median (see pricing.py), so a
# margin derived from it would launder the same randomness into something that
# merely looked principled.

POPULARITY_SCALE = 1000.0
POPULARITY_MAX = 10000

BUCKET_BASE_MARGIN_PCT: Dict[str, int] = {
    "bakery": 42,
    "snacks_sweets": 38,
    "coffee_tea": 35,
    "produce": 35,
    "beverages_soft": 30,
    "condiments_sauces": 30,
    "meals_chilled_frozen": 28,
    "sweeteners_syrups": 28,
    "default": 25,
    "oils_fats": 24,
    "olive_oil": 22,
    "dairy": 20,
}

LABEL_MARGIN_UPLIFT: Dict[str, float] = {
    "en:organic": 1.25,
    "en:usda-organic": 1.25,
    "en:eu-organic": 1.25,
    "en:fair-trade": 1.15,
    "en:rainforest-alliance": 1.15,
    "en:no-gluten": 1.10,
    "en:no-lactose": 1.10,
}

LABEL_UPLIFT_CAP = 1.40

MARGIN_SOURCE_STAMP = "modelled_category_margin"
POPULARITY_SOURCE_STAMP = "open_food_facts_unique_scans"


def derive_popularity(unique_scans_n: Optional[int]) -> int:
    """``1000 * ln(1 + unique_scans_n)``, capped at the 0..10000 envelope.

    One sentence explains it to a demo audience: every e-fold of real scanners
    is worth a thousand points, and the cap is reached at 22,026 scanners. Scan
    counts are heavy-tailed over four orders of magnitude, so a raw count would
    leave almost the whole catalogue at a value the envelope cannot tell from
    zero. Monotone, so it never reorders two products against their true counts.

    No scans derives 0, which is exactly the field map's ``missing: 0``: an
    unscanned product keeps its unmodified relevance score, never a demotion.
    """
    if not unique_scans_n or unique_scans_n <= 0:
        return 0
    return min(POPULARITY_MAX, round(POPULARITY_SCALE * math.log1p(unique_scans_n)))


def derive_margin(bucket_name: str, labels_tags: Optional[list]) -> Tuple[int, str]:
    """Return ``(margin_pct, breakdown)`` from the pricing bucket and real labels.

    The breakdown is stored on the product as ``Modelled margin`` so the
    derivation is legible in the record itself, the way the estimated unit price
    already ships its own debug string.
    """
    key = bucket_name if bucket_name in BUCKET_BASE_MARGIN_PCT else "default"
    base = BUCKET_BASE_MARGIN_PCT[key]
    uplift = 1.0
    matched: list[str] = []
    for tag in labels_tags or []:
        rate = LABEL_MARGIN_UPLIFT.get(tag)
        if rate is not None:
            uplift *= rate
            matched.append(tag[3:] if tag.startswith("en:") else tag)
    uplift = min(uplift, LABEL_UPLIFT_CAP)
    margin = round(base * uplift)
    detail = f"bucket={key} base={base}%"
    if matched:
        detail += f", labels={'+'.join(matched)} x{uplift:.2f}"
    return margin, f"{margin}% ({detail})"


# ----------------------------
# Attributes extraction (OFF -> attrs)
# ----------------------------

def join_tags(tags: Any, prefix_strip: Optional[str] = None, sep: str = ", ") -> Optional[str]:
    if not isinstance(tags, list):
        return None
    vals: list[str] = []
    for x in tags:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        if prefix_strip and s.startswith(prefix_strip):
            s = s[len(prefix_strip):]
        vals.append(s)
    if not vals:
        return None
    return sep.join(vals)


def get_first_str(product: Dict[str, Any], *keys: str) -> Optional[str]:
    for k in keys:
        v = product.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def format_nutrient(nutriments: Dict[str, Any], key_100g: str, unit_key: Optional[str] = None) -> Optional[str]:
    if not isinstance(nutriments, dict):
        return None
    v = nutriments.get(key_100g)
    if v is None or not isinstance(v, (int, float)):
        return None
    unit = nutriments.get(unit_key) if unit_key and isinstance(nutriments.get(unit_key), str) else None
    if unit:
        return f"{v:g} {unit}"
    return f"{v:g}"


def build_attrs(
    product: Dict[str, Any],
    primary_category_label: Optional[str],
) -> Dict[str, str]:
    attrs: Dict[str, str] = {}

    qty = get_first_str(product, "quantity")
    if qty:
        attrs["Quantity"] = qty

    serving = get_first_str(product, "serving_size")
    if serving:
        attrs["Serving size"] = serving

    nutri = get_first_str(product, "nutrition_grades", "nutriscore_grade")
    if nutri and nutri.lower() != "unknown":
        attrs["Nutri-Score"] = nutri.upper()

    nova = product.get("nova_group")
    if isinstance(nova, (int, float)):
        attrs["NOVA group"] = str(int(nova))

    eco = get_first_str(product, "ecoscore_grade", "environmental_score_grade")
    if eco and eco.lower() != "unknown":
        attrs["Eco-Score"] = eco.upper()

    allergens = join_tags(product.get("allergens_tags"), prefix_strip="en:")
    if allergens:
        attrs["Allergens"] = allergens

    labels = join_tags(product.get("labels_tags"), prefix_strip="en:")
    if labels:
        attrs["Labels"] = labels

    analysis = join_tags(product.get("ingredients_analysis_tags"), prefix_strip="en:")
    if analysis:
        attrs["Ingredients analysis"] = analysis

    countries = get_first_str(product, "countries")
    if countries:
        attrs["Countries"] = countries

    # Category: suppress undefined-like values.
    #
    # There is deliberately no second, tag-derived fallback here. One existed and
    # was unreachable: ``primary_category_label`` is the first entry of the flat
    # ``categories`` list, which already dropped undefined-like labels, so it is
    # absent only when *every* tag was undefined-like — exactly the case a
    # fallback would have to suppress too. Reviving it would also mean a second
    # rule for naming a category, which is the defect this field's labelling was
    # just single-sourced to remove.
    if primary_category_label and not _is_undefined_like(primary_category_label):
        attrs["Category"] = primary_category_label

    nutriments = product.get("nutriments")
    if isinstance(nutriments, dict):
        energy_kcal = format_nutrient(nutriments, "energy-kcal_100g", "energy-kcal_unit")
        if energy_kcal:
            attrs["Energy (kcal/100g)"] = energy_kcal
        fat = format_nutrient(nutriments, "fat_100g", "fat_unit")
        if fat:
            attrs["Fat (g/100g)"] = fat
        sat = format_nutrient(nutriments, "saturated-fat_100g", "saturated-fat_unit")
        if sat:
            attrs["Saturated fat (g/100g)"] = sat
        sugars = format_nutrient(nutriments, "sugars_100g", "sugars_unit")
        if sugars:
            attrs["Sugars (g/100g)"] = sugars
        salt = format_nutrient(nutriments, "salt_100g", "salt_unit")
        if salt:
            attrs["Salt (g/100g)"] = salt
        protein = format_nutrient(nutriments, "proteins_100g", "proteins_unit")
        if protein:
            attrs["Protein (g/100g)"] = protein
        fiber = format_nutrient(nutriments, "fiber_100g", "fiber_unit")
        if fiber:
            attrs["Fiber (g/100g)"] = fiber

    return attrs


def build_description(title: str, desc: str, attrs: Dict[str, str], *, single_line: bool = True) -> str:
    preferred_keys = [
        "Category", "Quantity", "Serving size", "Nutri-Score", "NOVA group", "Eco-Score",
        "Dietary restrictions",
        "Allergens", "Labels", "Ingredients analysis",
        "Energy (kcal/100g)", "Fat (g/100g)", "Saturated fat (g/100g)",
        "Sugars (g/100g)", "Salt (g/100g)", "Protein (g/100g)", "Fiber (g/100g)",
        "Countries"
    ]

    specs: list[str] = []
    for k in preferred_keys:
        v = attrs.get(k)
        if not v:
            continue
        if _is_undefined_like(v):
            continue
        specs.append(f"{k}: {v}")

    t = title.strip()
    d = desc.strip()

    # Avoid double periods ("Title..")
    if t.endswith("."):
        base = f"{t} {d}".strip()
    else:
        base = f"{t}. {d}".strip()

    if not specs:
        return base

    if single_line:
        return f"{base} Key specifications: " + "; ".join(specs)

    # Multiline (plain text, no markdown)
    lines = [t, "", d, "", "Key specifications:"]
    lines += [f"- {s}" for s in specs]
    return "\n".join(lines)


# ----------------------------
# Streaming parse
# ----------------------------

def iter_products(infile: TextIO) -> Iterator[Dict[str, Any]]:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            yield {"__bad_json__": True}
            continue
        if isinstance(obj, dict):
            yield obj


@dataclass
class Counters:
    read: int = 0
    written: int = 0
    bad_json: int = 0
    missing_code: int = 0
    missing_title: int = 0
    missing_desc: int = 0
    missing_image: int = 0
    missing_category: int = 0
    with_category_path: int = 0
    missing_category_path: int = 0
    unanchored_category_path: int = 0
    categories_at_multiple_addresses: int = 0
    categories_under_multiple_labels: int = 0
    labels_shared_by_multiple_categories: int = 0
    products_with_refused_category_tags: int = 0
    refused_category_tags: int = 0


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _progress_line(c: Counters, elapsed_s: float) -> str:
    rps = c.read / elapsed_s if elapsed_s > 0 else 0.0
    wps = c.written / elapsed_s if elapsed_s > 0 else 0.0
    return (
        f"Elapsed {elapsed_s:,.1f}s | "
        f"Read {_fmt_int(c.read)} ({rps:,.0f}/s) | "
        f"Wrote {_fmt_int(c.written)} ({wps:,.0f}/s) | "
        f"Skipped: title {_fmt_int(c.missing_title)}, desc {_fmt_int(c.missing_desc)}, "
        f"image {_fmt_int(c.missing_image)}, cat {_fmt_int(c.missing_category)}"
    )


# ----------------------------
# CLI
# ----------------------------

def build_parser(
    default_input: Path,
    default_output: Path,
    default_report: Path,
    default_pricing: Path,
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract Icecat-like NDJSON demo catalog from Open Food Facts JSONL/JSONL.GZ export."
    )
    p.add_argument("--input", type=Path, default=default_input)
    p.add_argument("--output", type=Path, default=default_output)
    p.add_argument("--report", type=Path, default=default_report)

    p.add_argument("--lang", default="en", help="Language code for titles, descriptions, and front images (e.g. en, fr, de).")

    p.add_argument("--require-category", action="store_true")
    p.add_argument(
        "--require-category-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop products whose hierarchical category_path does not resolve "
        "against the taxonomy, for a cleaner fully-faceted catalog (default: on). "
        "A path resolves when it is non-empty AND anchored to a global taxonomy "
        "root: a chain that starts mid-taxonomy was cut short by the language "
        "filter and files its categories at addresses no other catalog shares. "
        "Use --no-require-category-path to keep them. Automatically disabled when "
        "the taxonomy is unavailable (e.g. --no-taxonomy), which always emits an "
        "empty path.",
    )

    # Include en:undefined by default so "Undefined" does not become the primary category.
    p.add_argument("--category-exclude", default="en:null,en:unknown,en:undefined")

    p.add_argument("--pricing-config", type=Path, default=default_pricing, help="Path to pricing_buckets.json")

    # Category taxonomy: drives the hierarchical ``category_path`` field. When the
    # file is missing it is downloaded from the public Open Food Facts taxonomy.
    p.add_argument(
        "--taxonomy",
        type=Path,
        default=None,
        help="Path to the Open Food Facts categories taxonomy JSON "
        "(default: data/taxonomy/categories.json; downloaded if absent).",
    )
    p.add_argument(
        "--no-taxonomy",
        action="store_true",
        help="Disable hierarchical category_path extraction (emit it empty).",
    )

    # Debug/perf controls
    p.add_argument("--max-input-lines", type=int, default=0)
    p.add_argument("--max-output-records", type=int, default=0)

    # Progress controls
    p.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Emit a progress line every N records read (0 disables).",
    )
    p.add_argument(
        "--progress-seconds",
        type=float,
        default=5.0,
        help="Emit a progress line at least every N seconds (0 disables).",
    )

    p.add_argument(
        "--yes",
        action="store_true",
        help="Automatically confirm overwriting the output file.",
    )

    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    root = repo_root()
    print(f"Resolved repo root: {root}", file=sys.stderr, flush=True)

    default_input = root / "data" / "json_source" / "openfoodfacts-products.jsonl.gz"
    default_output = root / "data" / "products" / "off_common.ndjson"
    default_report = root / "data" / "products" / "report.json"
    default_pricing = root / "config" / "pricing_buckets.json"
    default_taxonomy = root / "data" / "taxonomy" / "categories.json"

    args = build_parser(default_input, default_output, default_report, default_pricing).parse_args(
        list(argv) if argv is not None else None
    )

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    if args.output.exists() and not args.yes:
        confirm = input(f"WARNING: Output file exists. Overwrite {args.output}? [y/N]: ").lower().strip()
        if confirm != "y":
            log("Aborted.")
            return 1

    if not args.input.exists():
        log(f"ERROR: input file not found: {args.input}")
        return 2

    if not args.pricing_config.exists():
        log(f"ERROR: pricing config not found: {args.pricing_config}")
        return 2

    pricing_cfg = load_pricing_config(args.pricing_config)

    taxonomy: Optional[Dict[str, Any]] = None
    if not args.no_taxonomy:
        taxonomy_path = args.taxonomy or default_taxonomy
        try:
            ensure_taxonomy(taxonomy_path, log=log)
            taxonomy = load_taxonomy(taxonomy_path)
            log(f"Category taxonomy: {len(taxonomy):,} nodes from {taxonomy_path}")
        except Exception as exc:  # noqa: BLE001 — degrade gracefully, never abort the run
            log(f"WARNING: could not load category taxonomy ({exc}); category_path will be empty.")
            taxonomy = None
    else:
        log("Category taxonomy disabled (--no-taxonomy); category_path will be empty.")

    # Resolve the clean-data gate. It only makes sense with a loaded taxonomy;
    # when there is none, emitting an empty path for every record means the gate
    # would drop the entire dataset, so disable it instead.
    require_category_path = args.require_category_path
    if require_category_path and taxonomy is None:
        log("Category_path gate disabled: taxonomy unavailable, so no product "
            "can resolve a path (keeping all records).")
        require_category_path = False

    ensure_parent_dir(args.output)
    ensure_parent_dir(args.report)

    c = Counters()
    t0 = time.time()
    last_progress_t = t0

    lang = args.lang.strip()
    cat_exclude = {x.strip() for x in args.category_exclude.split(",") if x.strip()}

    # One canonical parent per category, decided once for the whole run rather
    # than per product. This is what pins a category to a single address across
    # the catalog; see off_demo_extract.taxonomy for the selection rule.
    # Language-blind on purpose: the graph a chain walks is the whole taxonomy,
    # the same one for every catalog. Filtering it by language deleted the edges
    # through every filtered node too, which promoted 90 nodes to roots of an
    # English run's forest and truncated the chains beneath them.
    canonical_parents: Optional[Dict[str, Optional[str]]] = None
    if taxonomy is not None:
        canonical_parents = build_canonical_parent_map(taxonomy, exclude=cat_exclude)
        roots = sum(1 for parent in canonical_parents.values() if parent is None)
        log(
            f"Canonical category parents: {len(canonical_parents):,} nodes, "
            f"{roots:,} roots (fewest hops to a root; ties by canonical id)"
        )
    # The taxonomy's real roots. These two counts should now agree — the graph is
    # unfiltered, so its roots *are* the taxonomy's — and any gap is what
    # --category-exclude stranded, which is exactly what the category_path gate
    # refuses.
    taxonomy_roots: Set[str] = global_roots(taxonomy) if taxonomy is not None else set()
    if taxonomy is not None:
        log(
            f"Taxonomy roots: {len(taxonomy_roots):,} global "
            "(a chain must reach one of these to count as resolved)"
        )
    # The ids this catalog may *name*: the leaf a product is filed under and the
    # values of the flat ``categories`` field, which is where the language filter
    # belongs. Narrower than the graph above, which a path may walk through in
    # any language. ``None`` when no taxonomy was loaded.
    vocabulary: Optional[CategoryVocabulary] = None
    if taxonomy is not None:
        vocabulary = CategoryVocabulary.for_catalog(
            taxonomy, default_keep_prefixes(lang), cat_exclude
        )
        log(
            f"Category vocabulary: {len(vocabulary.eligible):,} ids this catalog "
            f"may file a product under (languages {sorted(default_keep_prefixes(lang))})"
        )

    address_audit = AddressAudit()
    tag_audit = TagCurationAudit()
    root_audit = RootAnchorAudit(
        taxonomy_roots=taxonomy_roots,
        traversal_roots={
            node for node, parent in (canonical_parents or {}).items() if parent is None
        },
    )

    log(f"Input:          {args.input}")
    log(f"Output:         {args.output}")
    log(f"Report:         {args.report}")
    log(f"Pricing config: {args.pricing_config}")
    log(f"Language:       {lang}")
    if args.require_category:
        log(f"Categories: require real category (exclude={sorted(cat_exclude)})")
    log("Starting extraction...")

    with open_maybe_gzip(args.input) as f, args.output.open("w", encoding="utf-8") as out:
        for product in iter_products(f):
            if args.max_input_lines and c.read >= args.max_input_lines:
                break

            c.read += 1

            if product.get("__bad_json__"):
                c.bad_json += 1
                continue

            code = str(product.get("code") or product.get("_id") or "").strip()
            if not code:
                c.missing_code += 1
                continue

            title = get_title(product, lang)
            if not title:
                c.missing_title += 1
                continue

            desc = get_description(product, lang)
            if not desc:
                c.missing_desc += 1
                continue

            image_url = compute_image_url(product, require_front_lang=lang)
            if not image_url:
                c.missing_image += 1
                continue

            # Alias renamed ids, refuse the ones that are not categories of this
            # catalog's taxonomy, and count what was refused. A refused *value*
            # never refuses its *record*: 19.6% of tagged products carry at least
            # one unresolvable tag, but only 2.5% have nothing else, so dropping
            # the record would throw away a clean lineage over one junk tag.
            tags_raw = extract_categories_tags(product)
            curated = curate_category_tags(tags_raw, vocabulary, cat_exclude)
            tag_audit.record(curated)
            if curated.rejected:
                c.products_with_refused_category_tags += 1
                c.refused_category_tags += len(curated.rejected)
            tags_curated = curated.accepted
            primary_tag = pick_primary_category_tag(tags_curated)

            if args.require_category and not primary_tag:
                c.missing_category += 1
                continue

            flat_entries = build_category_label_entries(
                primary_tag,
                tags_curated,
                taxonomy,
                lang,
                max_n=MAX_NUM_CATEGORIES,
                vocabulary=vocabulary,
            )
            categories = [label for _tag, label in flat_entries]
            primary_category_label = categories[0] if categories else None

            # Hierarchical, musgrave-style category path derived from the OFF
            # taxonomy graph (a single clean root→leaf chain as cumulative path
            # strings). The flat ``categories`` list above is still used for
            # pricing-bucket matching and attrs; ``category_path`` is the field
            # PRISM's hierarchical category facet renders.
            path_entries = (
                category_path_entries(
                    tags_curated,
                    taxonomy,
                    cat_exclude,
                    lang,
                    canonical_parents=canonical_parents,
                )
                if taxonomy is not None
                else []
            )
            category_path = [path for _node, path in path_entries]

            # A chain is walked to a root of *this run's* parent map, which is a
            # global taxonomy root unless --category-exclude stranded the head's
            # ancestry. When it did, the chain is truncated: it starts mid-
            # taxonomy at an address no other catalog agrees with. Expected to be
            # zero on a default run, and recorded for every record either way so
            # the zero is evidence rather than an absence of measurement.
            truncated_at = unanchored_head(
                [node for node, _path in path_entries], taxonomy_roots
            )
            if truncated_at is not None:
                root_audit.record(truncated_at)

            # Optional clean-data gate: skip products whose hierarchy doesn't
            # resolve. Done before the pricing model so dropped records are cheap.
            # "Resolve" means anchored, not merely non-empty — a path that starts
            # mid-taxonomy files its categories at addresses that exist nowhere
            # else, which is the defect the gate is supposed to promise against.
            if require_category_path and not category_path:
                c.missing_category_path += 1
                continue
            if require_category_path and truncated_at is not None:
                c.unanchored_category_path += 1
                continue

            attrs = build_attrs(product, primary_category_label=primary_category_label)

            dietary_restrictions = dietary_restrictions_from_off(product)
            if dietary_restrictions:
                attrs["Dietary restrictions"] = ", ".join(dietary_restrictions)

            labels_tags = product.get("labels_tags") if isinstance(product.get("labels_tags"), list) else []
            brand = product.get("brands") if isinstance(product.get("brands"), str) else ""
            quantity = product.get("quantity") if isinstance(product.get("quantity"), str) else None
            serving_size = product.get("serving_size") if isinstance(product.get("serving_size"), str) else None

            price, bucket_name, unit_debug = estimate_price(
                gtin=pad_gtin13(code),
                primary_category=primary_category_label or "",
                categories=categories,
                quantity=quantity,
                serving_size=serving_size,
                labels_tags=labels_tags,
                brand=brand,
                config=pricing_cfg,
                title=title, 
            )

            attrs["Price source"] = "estimated_unit_model"
            attrs["Pricing bucket"] = bucket_name
            attrs["Estimated unit price"] = unit_debug

            gtin = pad_gtin13(code)

            # Business-signal values, with their provenance recorded next to the
            # existing "Price source" stamp so a demo viewer can tell the
            # observed signal from the modelled one (elastic/prism#5027).
            unique_scans_n = product.get("unique_scans_n")
            popularity = derive_popularity(unique_scans_n)
            margin, margin_detail = derive_margin(bucket_name, labels_tags)
            attrs["Margin source"] = MARGIN_SOURCE_STAMP
            attrs["Modelled margin"] = margin_detail
            attrs["Popularity source"] = POPULARITY_SOURCE_STAMP
            attrs["Unique scans (Open Food Facts)"] = str(unique_scans_n or 0)

            attr_keys = sorted(attrs.keys())
            description = build_description(title=title, desc=desc, attrs=attrs, single_line=True)

            doc = {
                "id": gtin,
                "title": title,
                "brand": brand or "",
                "description": description,
                "image_url": image_url,
                "price": price,
                "margin": margin,
                "popularity": popularity,
                "currency": pricing_cfg.currency,
                "categories": categories,
                "category_path": category_path,
                "attrs": attrs,
                "attr_keys": attr_keys,
                "dietary_restrictions": dietary_restrictions,
            }

            if category_path:
                c.with_category_path += 1
            # Property 2 (one address per category) and the label invariants
            # (one label per category, one category per label), on the records
            # actually written. The flat entries go in even when no path
            # resolved, so a tag that never reaches a chain is still audited.
            address_audit.record(path_entries, flat_entries)

            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            c.written += 1

            if args.max_output_records and c.written >= args.max_output_records:
                break

            now = time.time()

            # Progress by count
            if args.progress_every and (c.read % args.progress_every == 0):
                elapsed = now - t0
                log(_progress_line(c, elapsed))
                last_progress_t = now

            # Progress by wall-clock seconds (helps even when progress_every is large)
            if args.progress_seconds and (now - last_progress_t >= args.progress_seconds):
                elapsed = now - t0
                log(_progress_line(c, elapsed))
                last_progress_t = now

    elapsed = time.time() - t0
    log(_progress_line(c, elapsed))

    for line in tag_audit.log_lines():
        log(line)
    for line in root_audit.log_lines(dropped=require_category_path):
        log(line)

    c.categories_at_multiple_addresses = address_audit.conflict_count
    c.categories_under_multiple_labels = address_audit.label_conflict_count
    c.labels_shared_by_multiple_categories = address_audit.shared_label_count
    audit_summary = address_audit.summary()
    if address_audit.conflict_count:
        log(
            f"WARNING: {address_audit.conflict_count:,} categories resolved to more "
            "than one path address in this run — category_path is no longer a strict "
            "tree. See category_path_addresses in the report."
        )
        for example in audit_summary["examples"]:
            log(f"  {example['category']}: {' | '.join(example['addresses'])}")
    if address_audit.label_conflict_count:
        log(
            f"WARNING: {address_audit.label_conflict_count:,} categories rendered "
            "under more than one label in this run — category_path and categories "
            "can no longer be joined on string. See category_path_addresses in the "
            "report."
        )
        for example in audit_summary["label_examples"]:
            log(f"  {example['category']}: {' | '.join(example['labels'])}")
    if address_audit.shared_label_count:
        log(
            f"WARNING: {address_audit.shared_label_count:,} labels are shared by "
            "more than one category in this run — joining categories to a "
            "category_path segment on string is ambiguous for those. See "
            "category_path_addresses in the report."
        )
        for example in audit_summary["shared_label_examples"]:
            log(f"  {example['label']}: {' | '.join(example['categories'])}")
    log("Done.")

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "pricing_config": str(args.pricing_config),
        "elapsed_seconds": elapsed,
        "counters": c.__dict__,
        # Property 2: every category must occupy exactly one position in the
        # tree, and render under exactly one label (and that label must name only
        # it). Reported per run so a regression shows up here rather than in a
        # hand audit of the built index.
        "category_path_addresses": audit_summary,
        # Every product tag that did not survive curation, by reason, with the
        # worst offenders named. Separate from the address audit above on
        # purpose: that one records where an *accepted* category landed, this one
        # records what never got in. Without it the unresolvable-tag rate is only
        # discoverable by reverse-mapping a built index against the taxonomy.
        "category_tag_curation": tag_audit.summary(),
        # Chains that resolved but stopped short of a global taxonomy root. The
        # gate refuses these; this is where a run that keeps them (or a run that
        # wants to know what it lost) reads the number and the offending heads.
        "category_path_anchoring": root_audit.summary(),
        "filters": {
            "lang": lang,
            "title": f"product_name_{lang} OR (lang=={lang} AND product_name)",
            "description": f"generic_name_{lang} OR ingredients_text_{lang} OR (lang=={lang} AND generic_name/ingredients_text)",
            "image": f"computed from images/front_{lang} + rev/imgid",
            "category": "required" if args.require_category else "optional",
            "category_tags": (
                "aliased through the curated rename map, then refused unless the "
                "id is a taxonomy node this catalog may file a product under; refused "
                "values are dropped, never their records"
                if taxonomy is not None
                else "no taxonomy loaded: tags aliased and curated-dropped only, not validated"
            ),
            "category_path": (
                "disabled"
                if args.no_taxonomy
                else f"hierarchical root->leaf path from OFF taxonomy "
                f"(written for {c.with_category_path:,}/{c.written:,} records"
                + (
                    f"; {c.missing_category_path:,} unresolved products dropped"
                    f", {c.unanchored_category_path:,} more dropped for a path "
                    "that stops short of a taxonomy root"
                    if require_category_path
                    else ""
                )
                + ")"
            ),
            "price": "category baseline unit model + deterministic noise + label premiums + retail rounding",
            "dietary_restrictions": "keyword list derived from labels_tags and ingredients_analysis_tags (positive-only)",
            "progress": f"every {args.progress_every} records and/or {args.progress_seconds}s",
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())