from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Set, TextIO

from off_demo_extract.pricing import load_pricing_config, estimate_price
from off_demo_extract.describe import DescribeConfig, generate_description, generate_search_text


IMAGE_BASE = "https://images.openfoodfacts.org/images/products"


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
# English-only fields
# ----------------------------

def get_english_title(product: Dict[str, Any]) -> Optional[str]:
    t = product.get("product_name_en")
    if isinstance(t, str) and t.strip():
        return t.strip()

    lang = product.get("lang") or product.get("lc")
    if lang == "en":
        t2 = product.get("product_name")
        if isinstance(t2, str) and t2.strip():
            return t2.strip()

    return None


def get_english_description(product: Dict[str, Any], max_len: int = 600) -> Optional[str]:
    for k in ("generic_name_en", "ingredients_text_en"):
        v = product.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:max_len]

    lang = product.get("lang") or product.get("lc")
    if lang == "en":
        for k in ("generic_name", "ingredients_text"):
            v = product.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()[:max_len]

    return None


def get_best_ingredients_text(product: Dict[str, Any], max_len: int = 600) -> Optional[str]:
    """
    Prefer English ingredients text if present, else fall back to non-language-specific.
    This is used for keyword extraction and fallback description generation.
    """
    for k in ("ingredients_text_en", "ingredients_text"):
        v = product.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()[:max_len]
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


def filter_category_tags(tags: list[str], exclude: Set[str]) -> list[str]:
    return [t for t in tags if t not in exclude]


def pick_primary_category_tag(tags: list[str], exclude: Set[str]) -> Optional[str]:
    for t in tags:
        if t not in exclude:
            return t
    return None


def prettify_category(tag: str) -> str:
    t = tag
    if ":" in t:
        t = t.split(":", 1)[1]
    t = t.replace("-", " ").replace("_", " ").strip()
    if not t:
        return tag
    return t[0].upper() + t[1:]


def build_categories_list(primary_tag: Optional[str], tags_filtered: list[str], max_n: int = 3) -> list[str]:
    seen: Set[str] = set()
    out: list[str] = []

    def add(tag: str) -> None:
        label = prettify_category(tag)
        if label not in seen:
            seen.add(label)
            out.append(label)

    if primary_tag:
        add(primary_tag)

    for t in tags_filtered:
        if len(out) >= max_n:
            break
        add(t)

    return out


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
    primary_category_tag: Optional[str],
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

    if primary_category_label:
        attrs["Category"] = primary_category_label
    elif primary_category_tag:
        attrs["Category"] = prettify_category(primary_category_tag)

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
    missing_title_en: int = 0
    missing_desc_en: int = 0
    missing_image: int = 0
    missing_category: int = 0


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _progress_line(c: Counters, elapsed_s: float) -> str:
    rps = c.read / elapsed_s if elapsed_s > 0 else 0.0
    wps = c.written / elapsed_s if elapsed_s > 0 else 0.0
    return (
        f"Elapsed {elapsed_s:,.1f}s | "
        f"Read {_fmt_int(c.read)} ({rps:,.0f}/s) | "
        f"Wrote {_fmt_int(c.written)} ({wps:,.0f}/s) | "
        f"Skipped: title {_fmt_int(c.missing_title_en)}, desc {_fmt_int(c.missing_desc_en)}, "
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
        description="Extract demo-ready NDJSON catalog from Open Food Facts JSONL/JSONL.GZ export."
    )
    p.add_argument("--input", type=Path, default=default_input)
    p.add_argument("--output", type=Path, default=default_output)
    p.add_argument("--report", type=Path, default=default_report)

    p.add_argument("--prefer-lang", default="en")
    p.add_argument("--require-front-lang", default="")

    p.add_argument("--require-category", action="store_true")
    p.add_argument("--category-exclude", default="en:null,en:unknown")

    p.add_argument("--pricing-config", type=Path, default=default_pricing, help="Path to pricing_buckets.json")

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

    # Describe config (optional phrase bank)
    phrase_bank_path = root / "config" / "phrase_bank.json"
    describe_cfg = DescribeConfig(
        phrase_bank_path=phrase_bank_path if phrase_bank_path.exists() else None,
        max_phrases=4,
        max_tags=4,
    )

    ensure_parent_dir(args.output)
    ensure_parent_dir(args.report)

    c = Counters()
    t0 = time.time()
    last_progress_t = t0

    req_front_lang = args.require_front_lang.strip() or None
    cat_exclude = {x.strip() for x in args.category_exclude.split(",") if x.strip()}

    log(f"Input:          {args.input}")
    log(f"Output:         {args.output}")
    log(f"Report:         {args.report}")
    log(f"Pricing config: {args.pricing_config}")
    if describe_cfg.phrase_bank_path:
        log(f"Phrase bank:    {describe_cfg.phrase_bank_path}")
    if req_front_lang:
        log(f"Images: require front_{req_front_lang}")
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

            title = get_english_title(product)
            if not title:
                c.missing_title_en += 1
                continue

            # Keep an English-ish fallback text for keywords / fallback descriptions
            ingredients_text = get_best_ingredients_text(product, max_len=600)

            # Keep existing "generic/ingredients" description as a fallback if needed
            fallback_desc = get_english_description(product)
            if not fallback_desc:
                c.missing_desc_en += 1
                continue

            image_url = compute_image_url(product, prefer_lang=args.prefer_lang, require_front_lang=req_front_lang)
            if not image_url:
                c.missing_image += 1
                continue

            tags_raw = extract_categories_tags(product)
            tags_filtered = filter_category_tags(tags_raw, cat_exclude)
            primary_tag = pick_primary_category_tag(tags_raw, cat_exclude)

            if args.require_category and not primary_tag:
                c.missing_category += 1
                continue

            categories = build_categories_list(primary_tag, tags_filtered, max_n=3)
            primary_category_label = categories[0] if categories else None

            brand = product.get("brands") if isinstance(product.get("brands"), str) else ""
            dietary_restrictions = dietary_restrictions_from_off(product)

            attrs = build_attrs(product, primary_category_tag=primary_tag, primary_category_label=primary_category_label)
            if dietary_restrictions:
                # Keep attrs display-friendly; parsed array lives in top-level field.
                attrs["Dietary restrictions"] = ", ".join(dietary_restrictions)

            # Price estimation (existing)
            labels_tags = product.get("labels_tags") if isinstance(product.get("labels_tags"), list) else []
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
            )

            attrs["Price source"] = "estimated_unit_model"
            attrs["Pricing bucket"] = bucket_name
            attrs["Estimated unit price"] = unit_debug

            # -------- NEW: algorithmic demo-friendly description + search text --------
            # Prefer generating a readable description. Only fall back to ingredients/attrs if needed.
            description = generate_description(
                gtin=pad_gtin13(code),
                title=title,
                brand=brand or "",
                categories=categories,
                dietary_restrictions=dietary_restrictions,
                ingredients_text=ingredients_text or fallback_desc,
                attrs=attrs,
                cfg=describe_cfg,
            )

            search_text = generate_search_text(
                title=title,
                brand=brand or "",
                categories=categories,
                dietary_restrictions=dietary_restrictions,
                ingredients_text=ingredients_text or fallback_desc,
                attrs=attrs,
                cfg=describe_cfg,
            )
            # ------------------------------------------------------------------------

            attr_keys = sorted(attrs.keys())


            # Fold matchable terms into the single description field.
            # Keep it short to avoid spammy-looking cards.
            if search_text:
                description = f"{description}\n\nKeywords: {search_text}"

            doc = {
                "id": pad_gtin13(code),
                "title": title,
                "brand": brand or "",
                "description": description,
                "image_url": image_url,
                "price": price,
                "currency": pricing_cfg.currency,
                "categories": categories,
                "attrs": attrs,
                "attr_keys": attr_keys,
                "dietary_restrictions": dietary_restrictions,
}

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

            # Progress by wall-clock seconds
            if args.progress_seconds and (now - last_progress_t >= args.progress_seconds):
                elapsed = now - t0
                log(_progress_line(c, elapsed))
                last_progress_t = now

    elapsed = time.time() - t0
    log(_progress_line(c, elapsed))
    log("Done.")

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "pricing_config": str(args.pricing_config),
        "phrase_bank": str(describe_cfg.phrase_bank_path) if describe_cfg.phrase_bank_path else None,
        "elapsed_seconds": elapsed,
        "counters": c.__dict__,
        "filters": {
            "english_title": "product_name_en OR (lang == en AND product_name)",
            "fallback_text": "generic_name_en / ingredients_text_en (or lang == en fallbacks)",
            "image": f"computed from images/front_{req_front_lang} + rev/imgid" if req_front_lang else "computed from images/front_* + rev/imgid",
            "category": "required" if args.require_category else "optional",
            "price": "category baseline unit model + deterministic noise + label premiums + retail rounding",
            "description": "algorithmic demo-friendly generator; ingredients/spec lists only used as fallback",
            "search_text": "denormalized matching field from title/brand/categories/dietary/ingredients/attrs",
            "progress": f"every {args.progress_every} records and/or {args.progress_seconds}s",
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())