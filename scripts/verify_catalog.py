#!/usr/bin/env python3
"""Verify a built catalog NDJSON against the taxonomy snapshot it was built from.

The extractor's own report records what the *run* believed it emitted. This
script re-derives the same properties from the **artifact on disk**, so a claim
about a catalog does not rest on the code that wrote it. It reads only the
NDJSON and the pinned taxonomy file; it never contacts a cluster.

What it checks, per catalog:

* **Property 3 — exactly one chain per product.** ``category_path`` must be a
  single root->leaf chain rendered as cumulative ``/``-joined strings: element
  *i* is element *i-1* plus one segment. A union of parallel branches, a gap, or
  a repeat is a violation and is named.
* **Property 2 — every category at exactly one address.** A category (a path
  segment) must occur under one and only one cumulative address across the whole
  catalog. Two addresses for one category means ``category_path`` is not a tree.
* **Anchoring.** Every chain's first segment must be the display label of one of
  the taxonomy's global roots. This is the artifact-side reading of the
  extractor's ``category_path_anchoring`` block.
* **Vocabulary against the pinned snapshot.** Every ``categories`` value and
  every path segment must be a display label of some node in the snapshot. Values
  present in the catalog but not in the snapshot are reported — the offline half
  of "compare the indexed vocabulary against the pinned snapshot", checked
  against the file that will be indexed rather than after the fact.

Usage:
    python scripts/verify_catalog.py \\
        --ndjson data/products/off_en_v14.ndjson \\
        --taxonomy data/taxonomy/categories.json \\
        --lang en [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from off_demo_extract.taxonomy import (  # noqa: E402
    display_label,
    global_roots,
    load_taxonomy,
)

MAX_EXAMPLES = 5


def _chain_segments(path: List[str]) -> List[str]:
    """The segment each cumulative element adds, or raise on a broken chain.

    Derived by subtracting the previous element rather than by splitting on the
    separator: labels have the separator neutralised at render time, but reading
    the delta is what actually proves the elements nest.
    """
    segments: List[str] = []
    previous = ""
    for index, element in enumerate(path):
        if index == 0:
            if not element:
                raise ValueError("empty first element")
            segments.append(element)
        else:
            prefix = previous + "/"
            if not element.startswith(prefix):
                raise ValueError(f"element {index} does not extend its predecessor: {element!r}")
            tail = element[len(prefix) :]
            if not tail or "/" in tail:
                raise ValueError(f"element {index} adds {tail!r}, not exactly one segment")
            segments.append(tail)
        previous = element
    return segments


def verify(ndjson: Path, taxonomy_path: Path, lang: str) -> Dict[str, Any]:
    taxonomy = load_taxonomy(taxonomy_path)
    root_labels: Set[str] = {display_label(taxonomy, node, lang) for node in global_roots(taxonomy)}
    vocabulary: Set[str] = {display_label(taxonomy, node, lang) for node in taxonomy}

    records = 0
    with_path = 0
    empty_path = 0
    ids: Dict[str, int] = defaultdict(int)
    chain_violations: List[Dict[str, Any]] = []
    unanchored: List[Dict[str, Any]] = []
    addresses: Dict[str, Set[str]] = defaultdict(set)
    off_vocabulary: Dict[str, int] = defaultdict(int)
    depths: Dict[int, int] = defaultdict(int)

    with ndjson.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records += 1
            doc = json.loads(line)
            ids[str(doc.get("id") or "")] += 1
            path = doc.get("category_path") or []
            for value in doc.get("categories") or []:
                if value not in vocabulary:
                    off_vocabulary[value] += 1
            if not path:
                empty_path += 1
                continue
            with_path += 1
            depths[len(path)] += 1
            try:
                segments = _chain_segments(path)
            except ValueError as exc:
                if len(chain_violations) < MAX_EXAMPLES:
                    chain_violations.append({"id": doc.get("id"), "reason": str(exc), "path": path})
                else:
                    chain_violations.append({"id": doc.get("id"), "reason": str(exc)})
                continue
            if segments[0] not in root_labels:
                if len(unanchored) < MAX_EXAMPLES:
                    unanchored.append({"id": doc.get("id"), "head": segments[0]})
                else:
                    unanchored.append({"id": doc.get("id")})
            for segment, address in zip(segments, path):
                addresses[segment].add(address)
                if segment not in vocabulary:
                    off_vocabulary[segment] += 1

    multi_address = {seg: sorted(addr) for seg, addr in addresses.items() if len(addr) > 1}
    return {
        "ndjson": str(ndjson),
        "lang": lang,
        "records": records,
        "with_category_path": with_path,
        "empty_category_path": empty_path,
        # An index keyed by ``id`` holds one document per distinct id, so this is
        # the number a post-index ``_count`` has to match — not the record count.
        # A silent gap between the two is how a catalog and its index start
        # disagreeing without either side looking wrong.
        "distinct_ids": len(ids),
        "duplicate_id_instances": records - len(ids),
        "distinct_categories_in_paths": len(addresses),
        "property_3_single_chain_violations": len(chain_violations),
        "property_3_examples": chain_violations[:MAX_EXAMPLES],
        "property_2_categories_at_multiple_addresses": len(multi_address),
        "property_2_examples": [
            {"category": seg, "addresses": addr}
            for seg, addr in list(multi_address.items())[:MAX_EXAMPLES]
        ],
        "unanchored_chains": len(unanchored),
        "unanchored_examples": unanchored[:MAX_EXAMPLES],
        "taxonomy_root_labels": len(root_labels),
        "values_outside_pinned_snapshot": len(off_vocabulary),
        "value_instances_outside_pinned_snapshot": sum(off_vocabulary.values()),
        "top_values_outside_pinned_snapshot": sorted(
            off_vocabulary.items(), key=lambda kv: -kv[1]
        )[:MAX_EXAMPLES],
        "path_depth_histogram": {str(k): depths[k] for k in sorted(depths)},
    }


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndjson", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--json", type=Path, default=None, help="also write the result here")
    args = parser.parse_args(argv)

    result = verify(args.ndjson, args.taxonomy, args.lang)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)

    failures = (
        result["property_3_single_chain_violations"]
        + result["property_2_categories_at_multiple_addresses"]
        + result["unanchored_chains"]
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
