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

To get a clean, musgrave-style hierarchy (an array of cumulative path strings
like ``["Beverages", "Beverages/Hot beverages", "Beverages/Hot beverages/Teas"]``)
we need the taxonomy *graph* — the parent→child edges — and then walk a single
canonical chain from the product's most specific category up to a root.

The taxonomy is the public OFF file:
    https://static.openfoodfacts.org/data/taxonomies/categories.json

Its shape is ``{canonical_id: {"name": {lang: label, ...},
"parents": [canonical_id, ...], "children": [...]}}``.

Strategy
--------
The address of a category must not depend on which product you are looking at.
So the parent of every node is decided **once per run, globally**, over the whole
taxonomy — never per product:

1. :func:`build_canonical_parent_map` runs one BFS from the taxonomy's roots over
   the reversed ``parents`` edges. That gives every node its exact *fewest-hops*
   distance to a root, and picks each node's single canonical parent. The result
   is a spanning forest of the DAG: every non-root keeps exactly one parent, no
   node is orphaned, only redundant parent edges are dropped.
2. :func:`category_chain` then takes the product's own tags only to choose the
   **leaf**, and walks the canonical parent map from that leaf all the way to a
   **global** root — materialising ancestors the product never tagged.
3. Each canonical id maps to a display label (taxonomy ``name`` in the requested
   language, falling back to English, then ``xx``, then a prettified slug) and
   the chain is emitted as cumulative ``/``-joined paths.

Why global anchoring, and not just a DAG-to-tree projection
-----------------------------------------------------------
The older per-product walk induced the subgraph of the product's own tags and
stopped at a *local* root of that subgraph. Two products carrying the same node
therefore filed it at different addresses whenever one of them omitted an
intermediate tag: ``en:sodas`` has exactly one parent at every hop of its lineage
— zero DAG forks — and still landed at two different addresses. Choosing a
canonical parent alone fixes none of that class; anchoring the walk to a global
root does.

The two properties this module guarantees
-----------------------------------------
**Property 2 — one address per category.** A node's ancestry comes from the
run-wide canonical parent map, so every occurrence of a node resolves to the
identical path, on every product.

**Property 3 — one path per product.** :func:`category_chain` returns a single
root→leaf chain, so a product's ``category_path`` is one cumulative path and
never a union of branches.

The tie-break
-------------
2,545 of the 14,457 taxonomy nodes have more than one parent, and 1,070 of those
(42%) have several parents tied on fewest-hops depth — so the tie-break decides
nearly half the multi-parent cases and is load-bearing, not a detail. The rule is
**the lexicographically smallest canonical id wins**, applied identically to
canonical-parent selection and to leaf selection.

It is chosen for stability, which is what actually matters: if the rule moved
between taxonomy refreshes, path addresses would move with it and previously
authored policies would silently break. Lexicographic order depends only on the
*set* of tied ids, so re-ordering a node's ``parents`` list — the most common
kind of churn in the upstream file — cannot move an address. Measured on the
current taxonomy, all 2,545 multi-parent nodes already list their parents in
lexicographic order, so this rule also agrees with the upstream authored order on
every one of the 1,070 ties today, while being immune to that order changing.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, TextIO, Tuple

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


def _lang_filter(keep_prefixes: Optional[Set[str]]):
    """Predicate keeping only canonical ids whose language prefix is wanted.

    ``keep_prefixes`` (e.g. ``{"en"}``) keeps an English catalog from surfacing a
    French-only node like ``fr:pates-a-tartiner`` mid-path. English is the
    taxonomy's backbone language, so callers typically include ``"en"``.
    """
    if not keep_prefixes:
        return lambda canonical_id: True

    def _ok(canonical_id: str) -> bool:
        prefix = canonical_id.split(":", 1)[0] if ":" in canonical_id else ""
        return prefix in keep_prefixes

    return _ok


def eligible_nodes(
    taxonomy: Dict[str, Any],
    keep_prefixes: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
) -> Set[str]:
    """Taxonomy ids this run is allowed to place in a path."""
    lang_ok = _lang_filter(keep_prefixes)
    skip = exclude or set()
    return {n for n in taxonomy if n not in skip and lang_ok(n)}


def build_canonical_parent_map(
    taxonomy: Dict[str, Any],
    keep_prefixes: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
) -> Dict[str, Optional[str]]:
    """Pick one canonical parent for every taxonomy node, once, for the whole run.

    The OFF category taxonomy is a DAG: 2,545 of its 14,457 nodes have more than
    one parent. This collapses it to a spanning **forest** — every non-root keeps
    exactly one parent, the roots stay roots, and no node is orphaned; only
    redundant parent edges are dropped.

    Selection rule, in order:

    1. **Fewest hops to a root wins.** One BFS from the roots over the reversed
       ``parents`` edges gives every node its exact shortest distance to a root;
       a node's canonical parent is one of its parents sitting on such a shortest
       route.
    2. **On a tie, the lexicographically smallest canonical id wins.** See the
       module docstring for why this rule and not another: it is the one that
       does not move when the upstream file re-orders a ``parents`` list.

    Returns ``{canonical_id: parent_id_or_None}`` covering every eligible node;
    ``None`` marks a root. The mapping is acyclic by construction, because a
    node's canonical parent always has a strictly smaller depth than the node.

    Build this **once per run** and thread it into :func:`category_chain` —
    rebuilding it per product would be both slow and pointless, since its whole
    purpose is to be product-independent.
    """
    nodes = eligible_nodes(taxonomy, keep_prefixes=keep_prefixes, exclude=exclude)

    def parents_of(node: str) -> List[str]:
        raw = taxonomy.get(node, {}).get("parents")
        if not isinstance(raw, list):
            return []
        return [p for p in raw if isinstance(p, str) and p in nodes]

    parents: Dict[str, List[str]] = {n: parents_of(n) for n in nodes}
    children: Dict[str, List[str]] = {}
    for node, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(node)

    # Pass 1 — exact fewest-hops depth for every node, by BFS from the roots.
    depth: Dict[str, int] = {}
    frontier = deque(sorted(n for n in nodes if not parents[n]))
    for root in frontier:
        depth[root] = 0

    def _drain() -> None:
        while frontier:
            cur = frontier.popleft()
            for child in children.get(cur, ()):
                if child not in depth:
                    depth[child] = depth[cur] + 1
                    frontier.append(child)

    _drain()

    # Defensive: the current OFF taxonomy has no cycles, but a future refresh
    # could introduce one, and a cycle has no root to reach. Break each such
    # component at its lexicographically smallest node so the result stays total
    # and deterministic instead of silently dropping the whole component.
    unreached = sorted(nodes - depth.keys())
    while unreached:
        pseudo_root = unreached[0]
        depth[pseudo_root] = 0
        frontier.append(pseudo_root)
        _drain()
        unreached = [n for n in unreached if n not in depth]

    # Pass 2 — canonical parent, decided from the finished depths so the result
    # never depends on BFS visit order.
    canonical: Dict[str, Optional[str]] = {}
    for node in nodes:
        d = depth[node]
        # Only a strictly shallower parent may be canonical: that is what makes
        # the map acyclic, and it correctly leaves a cycle-broken pseudo-root
        # (depth 0) parentless.
        uphill = [p for p in parents[node] if depth[p] < d]
        canonical[node] = min(uphill, key=lambda p: (depth[p], p)) if uphill else None
    return canonical


def canonical_ancestry(
    canonical_parents: Dict[str, Optional[str]], node: str
) -> List[str]:
    """Root→``node`` chain of ids from the canonical parent map.

    Returns ``[]`` for a node the map does not cover.
    """
    if node not in canonical_parents:
        return []
    chain: List[str] = []
    seen: Set[str] = set()
    cur: Optional[str] = node
    while cur is not None and cur not in seen:
        chain.append(cur)
        seen.add(cur)
        cur = canonical_parents.get(cur)
    chain.reverse()
    return chain


def category_chain(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    keep_prefixes: Optional[Set[str]] = None,
    canonical_parents: Optional[Dict[str, Optional[str]]] = None,
) -> List[str]:
    """Return one canonical root→leaf chain of category ids for this product.

    The product's own tags choose only the **leaf**; the rest of the chain comes
    from the run-wide canonical parent map and runs all the way to a **global**
    taxonomy root, materialising ancestors this product never tagged. That is
    what makes a node's address identical on every product carrying it
    (property 2). Exactly one chain is returned (property 3). Returns ``[]``
    when the product has no taxonomy-known category.

    Leaf selection:

    1. Drop any tag that is a canonical ancestor of another of this product's
       tags — an ancestor is already on the more specific tag's chain.
    2. Of what remains, the **longest** canonical chain wins (the most specific
       filing), and **on a tie the lexicographically smallest canonical id
       wins** — the same tie-break the parent map uses.

    ``canonical_parents`` should be the map built once per run by
    :func:`build_canonical_parent_map`. It is optional only so ad-hoc callers and
    tests can pass tags alone; omitting it rebuilds the map on every call, which
    is far too slow for an extraction run. When it *is* supplied, the language
    filter is already baked into it and ``keep_prefixes`` is not consulted again.
    """
    if canonical_parents is None:
        canonical_parents = build_canonical_parent_map(
            taxonomy, keep_prefixes=keep_prefixes, exclude=exclude
        )

    present: Set[str] = {
        t for t in product_tags if t not in exclude and t in canonical_parents
    }
    if not present:
        return []

    chains: Dict[str, List[str]] = {
        node: canonical_ancestry(canonical_parents, node) for node in present
    }

    # A tag that lies on another tag's chain is redundant: the deeper tag already
    # carries it. Without this, a product tagged both a node and a descendant of
    # it that took a shortcut edge could file under the shallower one.
    covered: Set[str] = set()
    for node, chain in chains.items():
        covered.update(chain[:-1])
    candidates = [n for n in present if n not in covered] or sorted(present)

    leaf = min(candidates, key=lambda n: (-len(chains[n]), n))
    return chains[leaf]


def category_path_entries(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    lang: str = "en",
    canonical_parents: Optional[Dict[str, Optional[str]]] = None,
) -> List[Tuple[str, str]]:
    """``(canonical_id, cumulative_path)`` for each step of the product's chain.

    Callers that only need the paths want :func:`build_category_path`; the ids
    are here so an extraction run can verify property 2 — that a node resolved to
    the same address everywhere — instead of hand-auditing the built index.
    """
    keep_prefixes = default_keep_prefixes(lang)
    chain = category_chain(
        product_tags,
        taxonomy,
        exclude,
        keep_prefixes=keep_prefixes,
        canonical_parents=canonical_parents,
    )

    entries: List[Tuple[str, str]] = []
    prefix = ""
    for node in chain:
        label = _display_name(taxonomy, node, lang)
        prefix = label if not prefix else f"{prefix}{PATH_SEPARATOR}{label}"
        entries.append((node, prefix))
    return entries


def default_keep_prefixes(lang: str) -> Set[str]:
    """Taxonomy languages a catalog in ``lang`` may put in a path.

    English is the taxonomy backbone; the persona language rides alongside it so
    localized-only nodes still resolve while foreign-language noise stays out.
    """
    return {"en", lang} if lang else {"en"}


def build_category_path(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    lang: str = "en",
    canonical_parents: Optional[Dict[str, Optional[str]]] = None,
) -> List[str]:
    """Cumulative root→leaf path strings, musgrave-style.

    Example: ``["Beverages", "Beverages/Hot beverages",
    "Beverages/Hot beverages/Teas", "Beverages/Hot beverages/Teas/Tea bags"]``.
    """
    return [
        path
        for _node, path in category_path_entries(
            product_tags, taxonomy, exclude, lang, canonical_parents=canonical_parents
        )
    ]


class AddressAudit:
    """Records where each category node landed, so a run can prove property 2.

    Property 2 — one address per category — holds by construction once the chain
    walks a run-wide canonical parent map, but "by construction" is exactly the
    kind of claim that quietly stops being true. Feeding every emitted chain
    through here turns a regression into a line in the extraction report rather
    than a hand audit of the built index.
    """

    def __init__(self) -> None:
        self.address: Dict[str, str] = {}
        self.conflicts: Dict[str, Set[str]] = {}

    def record(self, entries: Iterable[Tuple[str, str]]) -> None:
        for node, path in entries:
            seen = self.address.setdefault(node, path)
            if seen != path:
                self.conflicts.setdefault(node, {seen}).add(path)

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)

    def summary(self, max_examples: int = 5) -> Dict[str, Any]:
        examples = [
            {"category": node, "addresses": sorted(paths)}
            for node, paths in sorted(self.conflicts.items())[:max_examples]
        ]
        return {
            "categories_seen": len(self.address),
            "categories_at_multiple_addresses": self.conflict_count,
            "examples": examples,
        }


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def open_text(path: Path) -> TextIO:  # small convenience used by tests
    return path.open("r", encoding="utf-8")
