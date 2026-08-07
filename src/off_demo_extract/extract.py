from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Set, TextIO, Tuple, Union

from off_demo_extract.category_tags import (
    CategoryVocabulary,
    TagCurationAudit,
    curate_category_tags,
)
from off_demo_extract.pricing import load_pricing_config, estimate_price
from off_demo_extract.taxonomy import (
    PINNED_TAXONOMY_SHA256,
    TaxonomySnapshotError,
    resolve_taxonomy,
    load_taxonomy,
    build_canonical_parent_map,
    category_addresses,
    default_keep_prefixes,
    display_label,
    global_roots,
    unanchored_head,
    path_strings,
    AddressAudit,
    CategoryAddresses,
    AddressIndex,
    RootAnchorAudit,
)


IMAGE_BASE = "https://images.openfoodfacts.org/images/products"

# Values we should treat as "not meaningful" and avoid emitting in attrs/description.
_UNDEFINED_LIKE = {"undefined", "unknown", "null", "none", "n/a", "na", ""}

# How many *incidental* tags the flat ``taxonomy_tags`` field carries. It bounds
# the field for the very long tag lists Open Food Facts occasionally holds; it
# does not bound the product's own category chain, which
# :func:`select_category_label_entries` keeps whole (see that function).
#
# The number is a display-era default, not a storage limit: the field was
# introduced carrying 3 values and raised to 20 in "Increased the number of
# categories extracted" with no recorded reason, and nothing downstream reads it
# — PRISM maps ``taxonomy_tags`` as a ``terms`` facet field, which has no length
# rule. So it is kept because it is harmless and because an unbounded field has
# no ceiling at all, not because a measured cost forces it: over the first
# 200,000 records of the public export only 6 of 135,716 tagged products (0.004%)
# have more eligible tags than this, and removing the cap outright would add 205
# bytes to 10.5 MB of emitted tag payload (+0.002%).
MAX_NUM_TAXONOMY_TAGS = 20

# The type of a single ``attrs`` value.
#
# Most attributes are scalars and stay strings. The ones Open Food Facts
# supplies as a *list* are written as lists, because ``attrs`` is mapped
# ``flattened`` in Elasticsearch and ``flattened`` indexes each element of an
# array as its own keyword. Joining a list into one string before writing makes
# the *combination* the indexed key: ``{"term": {"attrs.Labels": "no-gluten"}}``
# then reaches only the products whose sole label is ``no-gluten``, and the term
# dictionary carries one entry per distinct combination instead of one per
# value.
#
# The union is written out rather than widened to ``Any`` on purpose: the
# distinction between "a list of two values" and "one value that happens to
# contain a comma" is the whole content of this field's shape, and four
# attributes (``Modelled margin``, ``Estimated unit price``, ``Serving size``,
# ``Quantity``) carry commas *inside* a single legitimate value. ``Any`` would
# stop the reader from seeing that there is a rule here at all.
AttrValue = Union[str, list[str]]


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
    the ``taxonomy_tags`` list would leave the same junk searchable by another route.
    """
    return tags[0] if tags else None


@dataclass
class FlatTagSelection:
    """What :func:`select_category_label_entries` emitted, and what it left out.

    The dropped labels are carried rather than counted so the run report can name
    them. A truncation nothing reports is invisible in the built catalog: the
    field is a flat list with no marker for "there was more", so a shortened list
    and a genuinely short one read identically.
    """

    entries: list[tuple[str, str]]
    eligible: int
    dropped: list[str]
    chain_over_cap: int = 0

    @property
    def truncated(self) -> bool:
        return bool(self.dropped)


class TagCapAudit:
    """Per-run totals for flat tag lists the cap shortened.

    Reported for every run, including the runs where it is zero — the same rule
    the refused-tag and unanchored-chain audits follow. The cap is the one place
    in the category pipeline that discards a *valid* value, and until #14 it did
    so with nothing counting it.
    """

    def __init__(self, max_n: int = MAX_NUM_TAXONOMY_TAGS, top_n: int = 20) -> None:
        self.max_n = max_n
        self.top_n = top_n
        self.products = 0
        self.tags_dropped = 0
        self.max_eligible = 0
        self.products_with_chain_over_cap = 0
        self.chain_over_cap_examples: list[str] = []
        self.dropped_labels: Counter = Counter()

    def record(self, code: str, selection: FlatTagSelection) -> None:
        self.max_eligible = max(self.max_eligible, selection.eligible)
        if selection.truncated:
            self.products += 1
            self.tags_dropped += len(selection.dropped)
            self.dropped_labels.update(selection.dropped)
        if selection.chain_over_cap:
            self.products_with_chain_over_cap += 1
            if len(self.chain_over_cap_examples) < self.top_n:
                self.chain_over_cap_examples.append(code)

    def summary(self) -> Dict[str, Any]:
        return {
            "max_taxonomy_tags": self.max_n,
            "max_eligible_tags_seen": self.max_eligible,
            "products_truncated": self.products,
            "tags_dropped": self.tags_dropped,
            "top_dropped_labels": [
                {"label": label, "products": n}
                for label, n in self.dropped_labels.most_common(self.top_n)
            ],
            # Products whose own chain is longer than the cap, so keeping the
            # chain whole took the list past ``max_taxonomy_tags``. Zero over the
            # first 200,000 records of the public export (deepest chain: 9
            # nodes), and reported so a taxonomy that grows deeper says so.
            "products_with_chain_over_cap": self.products_with_chain_over_cap,
            "chain_over_cap_examples": list(self.chain_over_cap_examples),
        }

    def log_lines(self) -> list[str]:
        lines = [
            f"Flat tag cap: {self.products:,} products truncated at "
            f"{self.max_n} tags ({self.tags_dropped:,} labels dropped); "
            f"longest eligible list seen {self.max_eligible:,}"
        ]
        if self.products_with_chain_over_cap:
            lines.append(
                f"  {self.products_with_chain_over_cap:,} products carry a category "
                f"chain longer than the cap; their taxonomy_tags exceed "
                f"{self.max_n} so the chain stays whole "
                f"(e.g. {', '.join(self.chain_over_cap_examples[:5])})"
            )
        return lines


def build_category_label_entries(
    primary_tag: Optional[str],
    tags_filtered: list[str],
    taxonomy: Optional[Dict[str, Any]],
    lang: str = "en",
    max_n: Optional[int] = MAX_NUM_TAXONOMY_TAGS,
    vocabulary: Optional[CategoryVocabulary] = None,
) -> list[tuple[str, str]]:
    """``(tag_id, label)`` for the flat ``taxonomy_tags`` field, primary tag first.

    The label comes from :func:`off_demo_extract.taxonomy.display_label` — the
    same function that names a ``category_path`` segment. This field used to
    de-slug the tag id itself, which rendered the identical node under a
    different string in the two fields (``Plant based foods`` here versus
    ``Plant-based foods`` there) and made them unjoinable by string; deriving
    both from one function is what stops a change to labelling from reaching one
    field and not the other.

    Ids ride along so a run can audit the two fields against each other. Callers
    that only want the strings want :func:`build_taxonomy_tags_list`; callers
    that have a chain to protect from the cap want
    :func:`select_category_label_entries`, which is what the extraction run uses.
    ``max_n`` of ``None`` means no cap, which is how that function builds the
    eligible list it then selects from.

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
        # Suppress "Undefined"/"Unknown"/etc from appearing as taxonomy_tags.
        if _is_undefined_like(label):
            return
        if label not in seen:
            seen.add(label)
            out.append((tag, label))

    if primary_tag:
        add(primary_tag)

    for t in tags_filtered:
        if max_n is not None and len(out) >= max_n:
            break
        add(t)

    return out


def select_category_label_entries(
    primary_tag: Optional[str],
    tags_filtered: list[str],
    taxonomy: Optional[Dict[str, Any]],
    lang: str = "en",
    max_n: int = MAX_NUM_TAXONOMY_TAGS,
    vocabulary: Optional[CategoryVocabulary] = None,
    chain_tags: Optional[Set[str]] = None,
) -> FlatTagSelection:
    """Choose the flat field's values, keeping the product's own chain whole.

    The cap used to be applied by walking the tags in order and stopping at
    ``max_n``, which drops the *tail* of the list. Open Food Facts orders
    ``categories_tags`` roughly general-to-specific, so the tail is where a
    product's most specific tags are — including, for three products in the first
    200,000 records of the public export, a node on its own emitted
    ``category_path``: ``0036800388352`` lost ``Basmati rices``, ``0051933012707``
    and ``0078742086774`` lost ``Peas``. The chain still showed the segment; the
    flat field no longer carried it, so a label-to-segment join missed a node the
    product genuinely tagged and the miss looked exactly like a labelling
    divergence (#14).

    ``chain_tags`` are the ids on the product's emitted chain. Every eligible tag
    among them is kept whatever the cap says, and the cap then governs the
    remaining, *incidental* tags. That makes the post-#10 invariant — every
    self-tagged chain node appears verbatim in ``taxonomy_tags`` — hold by
    construction rather than by there happening to be few enough tags: raising
    the cap to 24 would have covered today's longest list and left the same
    defect waiting for a 25-tag product to arrive, silently.

    The primary tag is reserved on the same footing, because the field's first
    value is read back as the product's primary category label (``attrs`` and the
    generated description both carry it), so it cannot be a truncation casualty.

    **Selection changes; order does not.** The kept entries stay in the order the
    tags arrived, so ``entries[0]`` is still the primary tag's label and a
    consumer reading position is unaffected. Only *which* tags survive a
    truncation changes — for 3 of 135,716 tagged products in that sample.
    """
    reserved: Set[str] = set(chain_tags or ())
    if primary_tag:
        reserved.add(primary_tag)

    eligible = build_category_label_entries(
        primary_tag, tags_filtered, taxonomy, lang, max_n=None, vocabulary=vocabulary
    )
    if len(eligible) <= max_n:
        return FlatTagSelection(entries=eligible, eligible=len(eligible), dropped=[])

    kept_reserved = [entry for entry in eligible if entry[0] in reserved]
    budget = max_n - len(kept_reserved)

    entries: list[tuple[str, str]] = []
    dropped: list[str] = []
    for tag, label in eligible:
        if tag in reserved:
            entries.append((tag, label))
        elif budget > 0:
            budget -= 1
            entries.append((tag, label))
        else:
            dropped.append(label)

    return FlatTagSelection(
        entries=entries,
        eligible=len(eligible),
        dropped=dropped,
        chain_over_cap=max(0, len(kept_reserved) - max_n),
    )


def build_taxonomy_tags_list(
    primary_tag: Optional[str],
    tags_filtered: list[str],
    taxonomy: Optional[Dict[str, Any]] = None,
    lang: str = "en",
    max_n: int = MAX_NUM_TAXONOMY_TAGS,
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
# explained to anyone watching, because the number underneath was noise. The
# draw was diagnosed from its signature — its mean landed on the midpoint of the
# range on every published catalog, which is what a uniform fill looks like and
# what a real signal never does. The test suite reproduces that measurement.
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

def clean_tags(tags: Any, prefix_strip: Optional[str] = None) -> Optional[list[str]]:
    """Normalize an Open Food Facts tag list and return it **as a list**.

    This used to be ``join_tags`` and ended with ``sep.join(vals)``. The join is
    gone, and this call site is the only place it could ever have been undone:
    here the source is still a list, so "two values" and "one value containing a
    comma" are still distinguishable. Once joined, they are the same string, and
    splitting on ``", "`` downstream shreds the four attributes that legitimately
    carry a comma inside one value (see :data:`AttrValue`).

    Order and duplicates are preserved exactly as the source supplies them.
    De-duplicating or sorting here would be a change to what the attribute says,
    which this correction deliberately does not make.
    """
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
    return vals


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
) -> Dict[str, AttrValue]:
    """Build the ``attrs`` map.

    List-sourced attributes are written as lists; everything else is a string.
    The four attributes that carry a comma inside one legitimate value are all
    scalars and are built from scalar sources, so nothing here can split them:
    ``Quantity`` and ``Serving size`` come straight from :func:`get_first_str`
    below, and ``Estimated unit price`` and ``Modelled margin`` are debug strings
    composed in ``main``.
    """
    attrs: Dict[str, AttrValue] = {}

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

    allergens = clean_tags(product.get("allergens_tags"), prefix_strip="en:")
    if allergens:
        attrs["Allergens"] = allergens

    labels = clean_tags(product.get("labels_tags"), prefix_strip="en:")
    if labels:
        attrs["Labels"] = labels

    analysis = clean_tags(product.get("ingredients_analysis_tags"), prefix_strip="en:")
    if analysis:
        attrs["Ingredients analysis"] = analysis

    # Countries stays a scalar read from the free-text ``countries`` field, and
    # is deliberately NOT part of this correction.
    #
    # It looks like a sixth multi-valued attribute and it is not one *here*: this
    # call site never had a list to preserve. Open Food Facts publishes the
    # canonical list separately as ``countries_tags``; reading it instead would
    # not be undoing a join, it would be changing which source field the
    # attribute reads, and with it the displayed value ("United States" becomes
    # "united-states"). That is a decision about what the catalog shows, not a
    # shape correction, so it is tracked on its own in #50 rather than made in
    # passing here.
    #
    # What must never happen is the other option: splitting this value on commas.
    # It is prose written by whoever edited the product -- the dump carries
    # "France, United States", "Frankreich,Deutschland" and
    # "France,États-Unis,en:france" -- and a comma split is the same heuristic
    # that shreds ``Quantity`` and ``Serving size``.
    countries = get_first_str(product, "countries")
    if countries:
        attrs["Countries"] = countries

    # Category: suppress undefined-like values.
    #
    # There is deliberately no second, tag-derived fallback here. One existed and
    # was unreachable: ``primary_category_label`` is the first entry of the flat
    # ``taxonomy_tags`` list, which already dropped undefined-like labels, so it is
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


# The ``attrs`` entries that are also written as their own top-level field, and
# the field each one is written to. Source key -> emitted field name.
#
# Why these seven and not the rest: an attribute earns a field when a query would
# want to reach that *fact* exactly -- a label, an allergen, a country, an
# ingredients-analysis verdict, a grade. The merchandising internals (margin,
# pricing bucket, price/popularity provenance) and the nutrition numerics stay in
# ``attrs`` only: the first are retailer-private and belong nowhere near shopper
# recall, and the second are numbers rendered as text ("1.2775 g"), which no
# useful query matches.
#
# ``Category`` is deliberately absent. It is ``taxonomy_tags[0]`` by construction
# (``build_attrs`` writes ``primary_category_label``), and every ``category_path``
# segment already joins to ``taxonomy_tags`` on string, so a promoted field would
# be a third spelling of a fact this document already carries twice.
#
# ``Dietary restrictions`` is absent for the same reason: it is already the
# top-level ``dietary_restrictions`` field, derived from ``labels_tags`` and
# ``ingredients_analysis_tags`` -- the same two sources ``labels`` and
# ``ingredients_analysis`` below are read from. Promoting it would be a second
# spelling of a field that already exists.
#
# The values are written exactly as ``build_attrs`` read them: a list-sourced
# attribute stays a list, and ``Countries`` stays the free-text scalar it is read
# from (see :func:`build_attrs` and #50). Nothing here cleans, normalises or
# re-cases a value -- these fields are the source's, and a "tidier" copy of a
# value would disagree with the ``attrs`` entry it was taken from.
PROMOTED_ATTR_FIELDS: Dict[str, str] = {
    "Labels": "labels",
    "Allergens": "allergens",
    "Countries": "countries",
    "Ingredients analysis": "ingredients_analysis",
    # Data-only, and named as such here so the next reader does not promote them
    # further. The reason is in the corpus: Open Food Facts contributors enter
    # per-serving figures into the per-100g fields, and the grades are computed
    # from those. Six confectionery SKUs in this dump record ``Sugars = 0
    # g/100g`` and are therefore graded Nutri-Score A at ~94% sugar. Emitting the
    # value is honest -- it is what the source says. Displaying it as a health
    # claim, or faceting on it, would make an unreliable number more prominent
    # rather than less, and the lie would be ours rather than the source's.
    "Nutri-Score": "nutri_score",
    "Eco-Score": "eco_score",
    "NOVA group": "nova_group",
}


def promoted_attr_fields(attrs: Mapping[str, AttrValue]) -> Dict[str, AttrValue]:
    """Return the top-level fields promoted out of ``attrs``.

    A key absent from ``attrs`` is absent here too: the writer omits a field
    rather than emitting ``""`` or ``[]``, which is the same rule ``build_attrs``
    already follows. An empty string would index as a real (empty) keyword term
    and an empty list would claim the attribute was read and found empty, and
    neither is what "Open Food Facts does not carry this for this product" means.

    ``attrs`` keeps every promoted key. The blob is the inspection surface, and
    live readers target it by name -- a business signal filters on
    ``attrs.NOVA group``, a script derives the modelled margin from
    ``attrs.Labels``, and a retailer plugin iterates ``attrs.items()``
    generically. Removing a key here would silently change all three: no error,
    just a clause that stops matching.

    Lists are copied, so the emitted document never has two keys sharing one
    mutable object.
    """
    fields: Dict[str, AttrValue] = {}
    for key, field in PROMOTED_ATTR_FIELDS.items():
        value = attrs.get(key)
        if value is None:
            continue
        fields[field] = list(value) if isinstance(value, list) else value
    return fields


def build_description(title: str, desc: str) -> str:
    """The product's own prose: its title, then the source text.

    ``desc`` is ``generic_name_<lang>`` or ``ingredients_text_<lang>`` (see
    :func:`get_description`) -- retail text a person wrote about the product.

    It used to carry a ``Key specifications:`` run of eighteen ``attrs`` entries
    appended to that prose, and that block is gone. It was
    added to compensate for products whose source text is thin, and the tail it
    compensated for is far smaller than the cost: measured on 500 live catalog
    documents, 79% already carry a substantial ingredient list and only 1% carry
    nothing, while the block made up roughly three quarters of the median
    description. Because BM25 normalises by field length, that padding
    down-weighted the real text it was glued to, and its label words discriminate
    nothing -- ``description ~ "Allergens"`` and ``~ "Nutri-Score"`` each matched
    *every* document in the catalog.

    The facts are not lost: the ones worth reaching exactly are now their own
    fields (:data:`PROMOTED_ATTR_FIELDS`), and every attribute remains in
    ``attrs``.
    """
    t = title.strip()
    d = desc.strip()

    # Avoid double periods ("Title..")
    if t.endswith("."):
        return f"{t} {d}".strip()
    return f"{t}. {d}".strip()


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
    categories_at_multiple_primary_addresses: int = 0
    products_at_multiple_addresses: int = 0
    categories_under_multiple_labels: int = 0
    labels_shared_by_multiple_categories: int = 0
    products_with_refused_category_tags: int = 0
    refused_category_tags: int = 0
    products_with_truncated_taxonomy_tags: int = 0
    truncated_taxonomy_tags: int = 0


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

    # Category taxonomy: drives the hierarchical ``category_path`` field. A
    # missing snapshot is an error; it is never downloaded unless asked for.
    p.add_argument(
        "--taxonomy",
        type=Path,
        default=None,
        help="Path to the Open Food Facts categories taxonomy JSON "
        "(default: data/json_source/categories.json). A missing file is an error, "
        "not a download; see --fetch-taxonomy.",
    )
    p.add_argument(
        "--fetch-taxonomy",
        action="store_true",
        help="Download the taxonomy to --taxonomy when it is missing, instead of failing. "
        "Refreshing the snapshot is deliberate, so it is named here and lands in the "
        "build record rather than happening on a cache miss.",
    )
    p.add_argument(
        "--allow-unpinned-taxonomy",
        action="store_true",
        help="Build against the --taxonomy file whatever its sha256, instead of requiring "
        "the pinned snapshot. The run is then not comparable to a pinned build, which is "
        "why it has to be said on the command line.",
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
    default_taxonomy = root / "data" / "json_source" / "categories.json"

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
        # Resolving the snapshot is a gate, not a best-effort step: a run that
        # cannot get the taxonomy it was asked for must stop rather than write a
        # catalog whose category addresses came from somewhere else. Only the
        # *parsing* below keeps the old degrade-to-empty behaviour.
        try:
            resolve_taxonomy(
                taxonomy_path,
                fetch=args.fetch_taxonomy,
                expected_sha256=None if args.allow_unpinned_taxonomy else PINNED_TAXONOMY_SHA256,
                log=log,
            )
        except TaxonomySnapshotError as exc:
            log(f"ERROR: {exc}")
            return 2
        try:
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
    address_index: Optional[AddressIndex] = None
    if taxonomy is not None:
        canonical_parents = build_canonical_parent_map(taxonomy, exclude=cat_exclude)
        roots = sum(1 for parent in canonical_parents.values() if parent is None)
        log(
            f"Canonical category parents: {len(canonical_parents):,} nodes, "
            f"{roots:,} roots (fewest hops to a root; ties by canonical id)"
        )
        # Every root→node path in the DAG, primary first. Built once per run for
        # the same reason as the map above: a category's addresses must not depend
        # on which product you are looking at, and the alternates are enumerated
        # from the same global, language-blind graph as the primary.
        address_index = AddressIndex(taxonomy, canonical_parents, exclude=cat_exclude)
        log(
            f"Category addresses: {len(address_index):,} nodes, "
            f"{address_index.multi_address_nodes:,} at more than one address "
            f"(most for one node: {address_index.max_addresses:,})"
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
    # values of the flat ``taxonomy_tags`` field, which is where the language filter
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
    cap_audit = TagCapAudit(MAX_NUM_TAXONOMY_TAGS)
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

            # Hierarchical category path derived from the OFF taxonomy graph, in
            # the shape retail catalogs typically expose (a single clean root→leaf
            # chain as cumulative path strings). The flat ``taxonomy_tags`` list
            # below is still used for pricing-bucket matching and attrs;
            # ``category_path`` is the field PRISM's hierarchical category facet
            # renders.
            #
            # It is derived *before* the flat list because the flat list's cap
            # needs to know which tags are on this chain, so that truncating the
            # incidental ones cannot take a segment's flat counterpart with them
            # (#14).
            addressing = (
                category_addresses(
                    tags_curated,
                    taxonomy,
                    cat_exclude,
                    lang,
                    canonical_parents=canonical_parents,
                    address_index=address_index,
                )
                if taxonomy is not None
                else CategoryAddresses(primary=[], entries=[])
            )
            # The primary address alone — the breadcrumb a product page leads
            # with, and the list every gate and audit below is scoped to. It is
            # byte-identical to the whole ``category_path`` of a build made before
            # the alternates existed, which is what makes the rebuild's diff
            # auditable: a primary that moves is a defect, not noise.
            path_entries = addressing.primary
            category_path_primary = path_strings(path_entries)
            # The field itself: the union across every address, primary first.
            category_path = path_strings(addressing.entries)

            flat_selection = select_category_label_entries(
                primary_tag,
                tags_curated,
                taxonomy,
                lang,
                max_n=MAX_NUM_TAXONOMY_TAGS,
                vocabulary=vocabulary,
                # The PRIMARY chain's nodes, deliberately, not the union's. The
                # cap reserves a slot for every node here, so widening it to the
                # alternates would change which incidental tags survive and move
                # ``taxonomy_tags`` on products whose addressing is the only thing
                # that changed — the one field this restoration is supposed to
                # leave byte-identical alongside the primary.
                chain_tags={node for node, _path in path_entries},
            )
            flat_entries = flat_selection.entries
            taxonomy_tags = [label for _tag, label in flat_entries]
            primary_category_label = taxonomy_tags[0] if taxonomy_tags else None

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
                # Written as the list it already is. The derivation rules that
                # produce this list are untouched; only the shape it is stored in
                # changes, and the ``dietary_restrictions`` field below has always
                # carried the same list. Copied rather than aliased so the two
                # keys of the emitted document never share one mutable object.
                attrs["Dietary restrictions"] = list(dietary_restrictions)

            labels_tags = product.get("labels_tags") if isinstance(product.get("labels_tags"), list) else []
            brand = product.get("brands") if isinstance(product.get("brands"), str) else ""
            quantity = product.get("quantity") if isinstance(product.get("quantity"), str) else None
            serving_size = product.get("serving_size") if isinstance(product.get("serving_size"), str) else None

            price, bucket_name, unit_debug = estimate_price(
                gtin=pad_gtin13(code),
                primary_category=primary_category_label or "",
                categories=taxonomy_tags,
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
            # observed signal from the modelled one. Without the stamps the two
            # are indistinguishable in the index, and a modelled number gets
            # discussed as if it were measured.
            unique_scans_n = product.get("unique_scans_n")
            popularity = derive_popularity(unique_scans_n)
            margin, margin_detail = derive_margin(bucket_name, labels_tags)
            attrs["Margin source"] = MARGIN_SOURCE_STAMP
            attrs["Modelled margin"] = margin_detail
            attrs["Popularity source"] = POPULARITY_SOURCE_STAMP
            attrs["Unique scans (Open Food Facts)"] = str(unique_scans_n or 0)

            attr_keys = sorted(attrs.keys())
            description = build_description(title=title, desc=desc)
            # The per-fact fields. Built from ``attrs`` after every writer above
            # has run, so a key added later is promoted by adding it to
            # PROMOTED_ATTR_FIELDS and nowhere else.
            promoted = promoted_attr_fields(attrs)

            # ``taxonomy_tags`` is the flat, display-only tag set: the product's
            # own category tags, validated against the taxonomy and labelled. It
            # is deliberately NOT named ``tags``. PRISM's ingest
            # (``sources.py``, ``normalized_json_resolve_row``) reads a row's
            # ``tags`` key *in preference to* its ``dietary_restrictions`` key
            # and assigns the result to the dietary field, so a catalog emitting
            # ``tags`` would silently overwrite real dietary data with this
            # category tag set. Do not "simplify" this name back to ``tags``.
            # It is also not ``categories``: that is the hierarchy facet's name,
            # which ``category_path`` below actually sources.
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
                "taxonomy_tags": taxonomy_tags,
                "category_path": category_path,
                # The one address the product page leads with, kept as its own
                # field rather than left implicit in ``category_path``'s ordering.
                # A breadcrumb renderer must not have to know that "the primary is
                # the first N values" — the ordering of a multi-valued keyword
                # field is not a contract anything downstream should lean on, and
                # the union is exactly the shape from which the primary cannot be
                # re-derived (which address is primary is a property of the graph,
                # not of the strings).
                "category_path_primary": category_path_primary,
                "attrs": attrs,
                "attr_keys": attr_keys,
                "dietary_restrictions": dietary_restrictions,
                # Present only for the attributes this product actually carries;
                # see :func:`promoted_attr_fields`.
                **promoted,
            }

            if category_path:
                c.with_category_path += 1
            # Property 2 (one PRIMARY address per category) and the label invariants
            # (one label per category, one category per label), on the records
            # actually written. The flat entries go in even when no path
            # resolved, so a tag that never reaches a chain is still audited.
            address_audit.record(path_entries, flat_entries, addressing.entries)
            # What the cap discarded from the written record. Counted here, with
            # the address audit, so the number describes the catalog that was
            # emitted rather than the records a later gate dropped.
            cap_audit.record(gtin, flat_selection)

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
    for line in cap_audit.log_lines():
        log(line)
    for line in root_audit.log_lines(dropped=require_category_path):
        log(line)

    c.products_with_truncated_taxonomy_tags = cap_audit.products
    c.truncated_taxonomy_tags = cap_audit.tags_dropped
    c.categories_at_multiple_primary_addresses = address_audit.conflict_count
    c.products_at_multiple_addresses = address_audit.multi_address_products
    c.categories_under_multiple_labels = address_audit.label_conflict_count
    c.labels_shared_by_multiple_categories = address_audit.shared_label_count
    audit_summary = address_audit.summary()
    log(
        f"Category addressing: {address_audit.multi_address_products:,} of "
        f"{address_audit.products:,} written products sit at more than one address; "
        f"{address_audit.multi_address_category_count:,} categories are emitted at "
        f"more than one (most for one category: "
        f"{address_audit.max_addresses_for_a_category:,}). "
        f"{len(address_audit.distinct_paths):,} distinct category_path values, "
        f"{address_audit.mean_path_values:.3f} mean per product, "
        f"{address_audit.max_path_values:,} at most."
    )
    if address_audit.conflict_count:
        log(
            f"WARNING: {address_audit.conflict_count:,} categories resolved to more "
            "than one PRIMARY path address in this run. Alternate addresses are "
            "expected and counted above; a category whose *primary* moves between "
            "products is a defect, and the expected value here is zero. See "
            "category_path_addresses in the report."
        )
        for example in audit_summary["examples"]:
            log(f"  {example['category']}: {' | '.join(example['addresses'])}")
    if address_audit.label_conflict_count:
        log(
            f"WARNING: {address_audit.label_conflict_count:,} categories rendered "
            "under more than one label in this run — category_path and taxonomy_tags "
            "can no longer be joined on string. See category_path_addresses in the "
            "report."
        )
        for example in audit_summary["label_examples"]:
            log(f"  {example['category']}: {' | '.join(example['labels'])}")
    if address_audit.shared_label_count:
        log(
            f"WARNING: {address_audit.shared_label_count:,} labels are shared by "
            "more than one category in this run — joining taxonomy_tags to a "
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
        # Property 2: every category must have exactly one PRIMARY position, and
        # render under exactly one label (and that label must name only it).
        # Alternate addresses are the restored DAG and are counted here as a shape,
        # not a violation. Reported per run so a regression shows up here rather
        # than in a hand audit of the built index. The distinct-path and
        # mean/max-values numbers in this block are what a downstream
        # ``category_path`` aggregation has to be sized against.
        "category_path_addresses": audit_summary,
        # Every product tag that did not survive curation, by reason, with the
        # worst offenders named. Separate from the address audit above on
        # purpose: that one records where an *accepted* category landed, this one
        # records what never got in. Without it the unresolvable-tag rate is only
        # discoverable by reverse-mapping a built index against the taxonomy.
        "category_tag_curation": tag_audit.summary(),
        # What the flat field's cap discarded. A tag dropped here is valid — it
        # survived curation — so it appears nowhere else in this report, and the
        # emitted field has no marker distinguishing a shortened list from a
        # short one. The product's own chain is never what gets dropped (#14);
        # this block says how much else did.
        "taxonomy_tags_cap": cap_audit.summary(),
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
            # Recorded per run so a build report says which per-fact fields that
            # run could emit, without anyone reading the source to find out.
            "promoted_attr_fields": (
                "written as top-level fields, verbatim from the attrs entry named, "
                "and omitted on a product that carries no such attribute: "
                + "; ".join(f"{field} <- attrs[{key!r}]" for key, field in PROMOTED_ATTR_FIELDS.items())
            ),
            "progress": f"every {args.progress_every} records and/or {args.progress_seconds}s",
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())