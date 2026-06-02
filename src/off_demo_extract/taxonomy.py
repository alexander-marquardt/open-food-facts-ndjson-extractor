"""
Open Food Facts category *taxonomy* → a single clean hierarchical path.

Background
----------
Open Food Facts ships ``categories_tags`` / ``categories_hierarchy`` on every
product, but those are **not** a single root→leaf path. They are the flattened
*union of every ancestor category* drawn from the OFF category taxonomy, which
is a directed acyclic graph (a category can have several parents). Naively
joining them with ``/`` produces a nonsense path that mixes parallel roots and
sibling branches.

To get a clean, retailer-style hierarchy (an array of cumulative path strings
like ``["Beverages", "Beverages/Hot beverages", "Beverages/Hot beverages/Teas"]``)
we need the taxonomy *graph* — the parent→child edges — and then walk a single
canonical chain from the product's most specific category up to a root.

The taxonomy is the public OFF file:
    https://static.openfoodfacts.org/data/taxonomies/categories.json

Its shape is ``{canonical_id: {"name": {lang: label, ...},
"parents": [canonical_id, ...], "children": [...]}}``.

Strategy
--------
We never trust the global taxonomy to pick the product's path for us — a
product is filed under many leaves across the whole graph. Instead we induce the
subgraph of *only this product's own tags* and find one canonical chain through
it:

1. Keep the product's tags that exist in the taxonomy (drops noise / mixed
   languages that aren't real taxonomy nodes).
2. Within that set, compute each node's depth (longest distance to a node in the
   set that has no in-set parent — i.e. a local root).
3. The leaf is the deepest node with no in-set child. Ties broken
   deterministically so runs are reproducible.
4. Walk parents upward, at each step choosing the in-set parent that yields the
   longest chain, until a local root is reached.
5. Map each canonical id to a display label (taxonomy ``name`` in the requested
   language, falling back to English, then ``xx``, then a prettified slug) and
   emit cumulative ``/``-joined paths.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO

TAXONOMY_URL = "https://static.openfoodfacts.org/data/taxonomies/categories.json"

# Path separator used *inside* each emitted cumulative path string. Matches the
# ``path_separator`` configured in the PRISM field map.
PATH_SEPARATOR = "/"


def ensure_taxonomy(path: Path, log: Optional[Any] = None) -> Path:
    """Return ``path``, downloading the OFF taxonomy there first if it's missing."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if log:
        log(f"Category taxonomy not found at {path}; downloading from {TAXONOMY_URL} ...")
    with urllib.request.urlopen(TAXONOMY_URL) as resp:  # noqa: S310 (trusted OFF host)
        data = resp.read()
    path.write_bytes(data)
    if log:
        log(f"Saved category taxonomy ({len(data):,} bytes) to {path}")
    return path


def load_taxonomy(path: Path) -> Dict[str, Any]:
    """Load the OFF category taxonomy JSON into a ``{canonical_id: node}`` dict."""
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Taxonomy at {path} is not a JSON object")
    return obj


def _prettify_slug(canonical_id: str) -> str:
    """Fallback label for a canonical id with no usable taxonomy ``name``."""
    t = canonical_id
    if ":" in t:
        t = t.split(":", 1)[1]
    t = t.replace("-", " ").replace("_", " ").strip()
    if not t:
        return canonical_id
    return t[0].upper() + t[1:]


def _display_name(taxonomy: Dict[str, Any], canonical_id: str, lang: str) -> str:
    """Human label for a category, preferring ``lang`` then English then ``xx``."""
    node = taxonomy.get(canonical_id) or {}
    names = node.get("name") if isinstance(node.get("name"), dict) else {}
    for key in (lang, "en", "xx"):
        val = names.get(key)
        if isinstance(val, str) and val.strip():
            label = val.strip()
            break
    else:
        label = _prettify_slug(canonical_id)
    # A literal separator inside a label would corrupt the cumulative path, so
    # neutralise it (the PRISM field map splits paths on PATH_SEPARATOR).
    return label.replace(PATH_SEPARATOR, " ")


def _in_set_parents(taxonomy: Dict[str, Any], node: str, present: Set[str]) -> List[str]:
    parents = taxonomy.get(node, {}).get("parents")
    if not isinstance(parents, list):
        return []
    return [p for p in parents if isinstance(p, str) and p in present]


def category_chain(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    keep_prefixes: Optional[Set[str]] = None,
) -> List[str]:
    """Return one canonical root→leaf chain of category ids for this product.

    Operates purely on the subgraph induced by the product's own taxonomy tags,
    so the result is faithful to how this product is actually filed rather than
    to the global graph. Returns ``[]`` when the product has no taxonomy-known
    category.

    ``keep_prefixes`` (e.g. ``{"en"}``) restricts the subgraph to canonical ids
    in those taxonomy languages, so an English catalog doesn't surface a
    French-only node like ``fr:pates-a-tartiner`` mid-path. English is the
    taxonomy's backbone language, so callers typically include ``"en"``.
    """

    def _lang_ok(canonical_id: str) -> bool:
        if not keep_prefixes:
            return True
        prefix = canonical_id.split(":", 1)[0] if ":" in canonical_id else ""
        return prefix in keep_prefixes

    present: Set[str] = {
        t
        for t in product_tags
        if t not in exclude and t in taxonomy and _lang_ok(t)
    }
    if not present:
        return []

    # Longest distance to a local root, memoised with a cycle guard.
    depth: Dict[str, int] = {}

    def get_depth(node: str, stack: Set[str]) -> int:
        if node in depth:
            return depth[node]
        best = 0
        for parent in _in_set_parents(taxonomy, node, present):
            if parent in stack:  # defensive: OFF taxonomy can contain cycles
                continue
            best = max(best, 1 + get_depth(parent, stack | {node}))
        depth[node] = best
        return best

    for node in present:
        get_depth(node, set())

    # Local roots are parents-of-something; anything that is never an in-set
    # parent is a leaf candidate.
    has_in_set_child: Set[str] = set()
    for node in present:
        for parent in _in_set_parents(taxonomy, node, present):
            has_in_set_child.add(parent)
    leaves = [n for n in present if n not in has_in_set_child]
    if not leaves:
        # Fully cyclic subgraph (degenerate); fall back to all nodes.
        leaves = sorted(present)

    # Deepest leaf wins; deterministic tie-break by canonical id.
    leaf = max(leaves, key=lambda n: (depth[n], n))

    chain: List[str] = [leaf]
    visited: Set[str] = {leaf}
    cur = leaf
    while True:
        parents = [
            p for p in _in_set_parents(taxonomy, cur, present) if p not in visited
        ]
        if not parents:
            break
        # Prefer the parent that extends the chain furthest toward a root.
        nxt = max(parents, key=lambda p: (depth[p], p))
        chain.append(nxt)
        visited.add(nxt)
        cur = nxt

    chain.reverse()  # root → leaf
    return chain


def build_category_path(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    lang: str = "en",
) -> List[str]:
    """Cumulative root→leaf path strings, retailer-style.

    Example: ``["Beverages", "Beverages/Hot beverages",
    "Beverages/Hot beverages/Teas", "Beverages/Hot beverages/Teas/Tea bags"]``.
    """
    # English is the taxonomy backbone; include the persona language alongside it
    # so localized-only nodes still resolve while foreign-language noise is kept
    # out of the path.
    keep_prefixes = {"en", lang} if lang else {"en"}
    chain = category_chain(product_tags, taxonomy, exclude, keep_prefixes=keep_prefixes)
    labels = [_display_name(taxonomy, node, lang) for node in chain]

    paths: List[str] = []
    prefix = ""
    for label in labels:
        prefix = label if not prefix else f"{prefix}{PATH_SEPARATOR}{label}"
        paths.append(prefix)
    return paths


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def open_text(path: Path) -> TextIO:  # small convenience used by tests
    return path.open("r", encoding="utf-8")
