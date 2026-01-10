from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Set, TextIO


IMAGE_BASE = "https://images.openfoodfacts.org/images/products"


# ----------------------------
# Repo / IO helpers
# ----------------------------

def repo_root() -> Path:
    """
    Find repo root by walking up from this file and looking for pyproject.toml.
    Falls back to current working directory if not found.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def open_maybe_gzip(path: Path, encoding: str = "utf-8") -> TextIO:
    """
    Open a .jsonl or .jsonl.gz file as a text stream (read-only).
    """
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding=encoding, errors="replace")
    return path.open("r", encoding=encoding, errors="replace")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------
# OFF image URL construction
# ----------------------------

def pad_gtin13(code: str) -> str:
    s = "".join(ch for ch in code if ch.isdigit())
    if len(s) >= 13:
        return s[:13]
    return ("0" * (13 - len(s))) + s


def product_folder_from_code(code: str) -> str:
    """
    OFF image folder scheme:
      - GTIN padded to 13 digits
      - split: 3/3/3/rest => aaa/bbb/ccc/rest
    Example: 0000101209159 -> 000/010/120/9159
    """
    c = pad_gtin13(code)
    return f"{c[0:3]}/{c[3:6]}/{c[6:9]}/{c[9:]}"


def pick_image_resolution_from_sizes(sizes: Dict[str, Any]) -> str:
    """
    Prefer 400 if available; else 200; else 100; else 'full'.
    """
    if not isinstance(sizes, dict):
        return "full"
    for res in ("400", "200", "100"):
        if res in sizes:
            return res
    return "full"


def build_selected_image_url(code: str, key: str, rev: str, sizes: Dict[str, Any]) -> str:
    """
    Selected image filename pattern:
      <key>.<rev>.<resolution>.jpg  (or .full.jpg)
    Example: front_en.3.400.jpg
    """
    res = pick_image_resolution_from_sizes(sizes)
    folder = product_folder_from_code(code)
    return f"{IMAGE_BASE}/{folder}/{key}.{rev}.{res}.jpg"


def build_raw_image_url(code: str, imgid: str, sizes: Dict[str, Any]) -> str:
    """
    Raw image filename pattern:
      <imgid>.<resolution>.jpg  (or <imgid>.jpg for full)
    Example: 1.400.jpg or 1.jpg
    """
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
    """
    Select a 'front' image key from OFF's `images` object.

    - If require_lang is set (e.g., 'en'), only accept front_<require_lang>.
    - Otherwise prefer front_<prefer_lang>, then any front_?? key, then 'front'.
    """
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
    """
    Compute a usable image URL from the OFF JSONL 'images' structure.

    Strategy:
    - pick a front_* key (prefer front_en unless require_front_lang is set)
    - if selected image has a 'rev', use selected URL
    - else if selected image has an 'imgid', use raw URL and locate sizes under images[imgid]
    """
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
    """
    English-only title.
    Prefer product_name_en. If product is marked as English, allow product_name.
    """
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
    """
    English-only description-like field.
    Prefer generic_name_en, then ingredients_text_en.
    If product is English, allow generic_name / ingredients_text.
    """
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


# ----------------------------
# Categories
# ----------------------------

def parse_category_exclude(csv: str) -> Set[str]:
    """
    Parse comma-separated excluded category tags.
    """
    items = [x.strip() for x in (csv or "").split(",")]
    return {x for x in items if x}


def extract_categories(product: Dict[str, Any]) -> list[str]:
    cats = product.get("categories_tags")
    if isinstance(cats, list):
        out: list[str] = []
        for x in cats:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def filter_categories(cats: list[str], exclude: Set[str]) -> list[str]:
    return [c for c in cats if c not in exclude]


def primary_category(cats: list[str], exclude: Set[str]) -> Optional[str]:
    for c in cats:
        if c not in exclude:
            return c
    return None


# ----------------------------
# Synthetic price
# ----------------------------

def synthetic_price(code: str, min_price: float, max_price: float) -> float:
    """
    Deterministic synthetic price from barcode.
    Produces a stable value in [min_price, max_price].
    """
    c = pad_gtin13(code)
    h = hashlib.blake2b(c.encode("utf-8"), digest_size=8).digest()
    n = int.from_bytes(h, "big")
    x = n / float(2**64 - 1)  # 0..1
    price = min_price + x * (max_price - min_price)
    return round(price, 2)


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


# ----------------------------
# CLI
# ----------------------------

def build_parser(default_input: Path, default_output: Path, default_report: Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract clean English NDJSON demo catalog from Open Food Facts JSONL/JSONL.GZ export."
    )
    p.add_argument("--input", type=Path, default=default_input, help=f"Input JSONL/JSONL.GZ (default: {default_input})")
    p.add_argument("--output", type=Path, default=default_output, help=f"Output NDJSON (default: {default_output})")
    p.add_argument("--report", type=Path, default=default_report, help=f"Run report JSON (default: {default_report})")

    p.add_argument("--prefer-lang", default="en", help="Preferred language for front image key (default: en)")
    p.add_argument(
        "--require-front-lang",
        default="",
        help="If set (e.g., 'en'), require front_<lang> image key; otherwise allow any front_*.",
    )

    p.add_argument(
        "--require-category",
        action="store_true",
        help="If set, require at least one non-placeholder category tag.",
    )
    p.add_argument(
        "--category-exclude",
        default="en:null,en:unknown",
        help="Comma-separated category tags to treat as placeholders and exclude (default: en:null,en:unknown).",
    )

    p.add_argument("--min-price", type=float, default=0.99)
    p.add_argument("--max-price", type=float, default=19.99)
    p.add_argument("--currency", type=str, default="EUR")

    # Debug/perf controls
    p.add_argument("--max-input-lines", type=int, default=0, help="Stop after reading N input lines (0 = no limit)")
    p.add_argument("--max-output-records", type=int, default=0, help="Stop after writing N clean records (0 = no limit)")
    p.add_argument("--progress-every", type=int, default=100000, help="Log progress every N read records")

    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    root = repo_root()
    default_input = root / "data" / "openfoodfacts-products.jsonl.gz"
    default_output = root / "out" / "off_en_clean.ndjson"
    default_report = root / "out" / "report.json"

    args = build_parser(default_input, default_output, default_report).parse_args(list(argv) if argv is not None else None)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    if not args.input.exists():
        log(f"ERROR: input file not found: {args.input}")
        log("Place the dataset under ./data/ or pass --input explicitly.")
        return 2

    ensure_parent_dir(args.output)
    ensure_parent_dir(args.report)

    c = Counters()
    t0 = time.time()

    req_front_lang = args.require_front_lang.strip() or None
    cat_exclude = parse_category_exclude(args.category_exclude)

    log(f"Input:  {args.input}")
    log(f"Output: {args.output}")
    log(f"Report: {args.report}")
    if req_front_lang:
        log(f"Images: require front_{req_front_lang}")
    if args.require_category:
        log(f"Categories: require real category (exclude={sorted(cat_exclude)})")

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

            desc = get_english_description(product)
            if not desc:
                c.missing_desc_en += 1
                continue

            image_url = compute_image_url(
                product,
                prefer_lang=args.prefer_lang,
                require_front_lang=req_front_lang,
            )
            if not image_url:
                c.missing_image += 1
                continue

            cats_raw = extract_categories(product)
            cats_filtered = filter_categories(cats_raw, cat_exclude)
            prim_cat = primary_category(cats_raw, cat_exclude)

            if args.require_category and not prim_cat:
                c.missing_category += 1
                continue

            price = synthetic_price(code=code, min_price=args.min_price, max_price=args.max_price)

            doc = {
                "id": pad_gtin13(code),
                "title": title,
                "description": desc,
                "image_url": image_url,
                "price": price,
                "currency": args.currency,
                "brand": product.get("brands"),
                # categories_tags are filtered to remove placeholders; primary_category is convenient for UI/facets
                "categories_tags": cats_filtered,
                "primary_category": prim_cat,
                "lang": "en",
            }

            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            c.written += 1

            if args.max_output_records and c.written >= args.max_output_records:
                break

            if args.progress_every and (c.read % args.progress_every == 0):
                elapsed = time.time() - t0
                log(f"Read {c.read:,} | Wrote {c.written:,} | Elapsed {elapsed:,.1f}s")

    elapsed = time.time() - t0
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "elapsed_seconds": elapsed,
        "counters": c.__dict__,
        "filters": {
            "english_title": "product_name_en OR (lang==en AND product_name)",
            "english_description": "generic_name_en OR ingredients_text_en OR (lang==en AND generic_name/ingredients_text)",
            "image": (
                f"computed from images/front_{req_front_lang} + rev/imgid"
                if req_front_lang
                else "computed from images/front_* + rev/imgid"
            ),
            "category": (
                f"require at least one category tag not in {sorted(cat_exclude)}"
                if args.require_category
                else f"kept (filtered placeholders: {sorted(cat_exclude)})"
            ),
            "price": f"synthetic deterministic [{args.min_price}, {args.max_price}] {args.currency}",
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())