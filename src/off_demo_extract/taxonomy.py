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
3. :func:`display_label` maps each canonical id to a display label (taxonomy
   ``name`` in the requested language, falling back to English, then ``xx``, then
   a prettified slug) and the chain is emitted as cumulative ``/``-joined paths.
   That function is the single place a category's label is decided — the flat
   ``categories`` field calls it too, so the two fields can never disagree about
   what one node is called.

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

What the language filter still cuts short
-----------------------------------------
Anchoring runs to a root of the map built for *this* run, and that map only
contains the languages the catalog may place in a path. 90 of the 161 roots an
English run sees are not taxonomy roots at all: they are nodes whose only parent
is foreign, so pruning promoted them. ``en:pate`` sits under
``fr:charcuteries-diverses`` under ``en:prepared-meats``, and an English catalog
therefore files it as a top-level ``Pâté``.

That residue is small. Over the first 300,000 records of the January 2026 dump,
455 products resolve an unanchored chain for English and 2 do for French; 6 of
those English ones also clear the extractor's title/description/image filters and
so reach the output, out of 16,847 records written. But it is the whole of what
"truncated" still means, and it is invisible from the path alone, which is why
:func:`global_roots` and :func:`unanchored_head` exist and why
``--require-category-path`` consults them rather than only testing the path for
emptiness. Removing the *cause* (teaching the walk to pass through a foreign
ancestor) is a separate change; refusing to emit an address that no other
catalog agrees with is this one.

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
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TextIO, Tuple

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


def display_label(taxonomy: Dict[str, Any], canonical_id: str, lang: str = "en") -> str:
    """The label a category node renders under — **the only** place that decides it.

    Every emitted field that shows a category to a human must call this: the
    hierarchical ``category_path`` segments and the flat ``categories`` values
    alike. Two renderings of the same node used to be produced by two rules — the
    taxonomy ``name`` here, and a mechanical de-slug of the tag id over in the
    extractor — so the same node read ``Plant-based foods`` in one field and
    ``Plant based foods`` in the other, and the two fields could not be joined on
    string (39% of products carried at least one such pair).

    Preference order is ``lang`` → English → ``xx`` → a prettified slug. The
    taxonomy ``name`` wins over the slug because it is the upstream-authored
    human label: it carries correct hyphenation (``Plant-based foods``),
    disambiguating parentheticals (``Crackers (Appetizers)``) and casing that a
    mechanical de-slug destroys — and, unlike the slug, it is **localised**, so a
    Spanish or French catalog renders Spanish or French rather than English.
    Every one of the taxonomy's 8,939 ``en:``-prefixed nodes has an ``en`` name,
    so on the English backbone the slug fallback never fires; it remains for
    localised-only nodes and for runs with no taxonomy at all (``--no-taxonomy``,
    which passes an empty mapping here).
    """
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


def parents_of(
    taxonomy: Dict[str, Any], node: str, within: Optional[Set[str]] = None
) -> List[str]:
    """A node's declared parents, restricted to ids that actually exist.

    ``within`` narrows the restriction further to a chosen subset — the eligible
    nodes of one run. Omitting it asks the question of the *whole* taxonomy,
    which is what tells a real root apart from one manufactured by pruning.
    """
    raw = taxonomy.get(node)
    raw = raw.get("parents") if isinstance(raw, dict) else None
    if not isinstance(raw, list):
        return []
    allowed: Any = taxonomy if within is None else within
    return [p for p in raw if isinstance(p, str) and p in allowed]


def global_roots(taxonomy: Dict[str, Any]) -> Set[str]:
    """The taxonomy's **true** roots: ids with no parent anywhere in the file.

    Deliberately blind to both the language filter and ``--category-exclude``,
    which is the whole point. :func:`build_canonical_parent_map` works over one
    run's *eligible* nodes, so a node whose only parent was pruned out of that
    set becomes a root **of the pruned map** while remaining a child in the
    taxonomy. Measured on the pinned 14,457-node snapshot: the taxonomy has 92
    roots, but the English-eligible map has 161, of which 90 are manufactured
    this way (``en:pate`` is one — it really sits under ``fr:charcuteries-
    diverses`` under ``en:prepared-meats``).

    A chain headed by such a node is *truncated*: its real ancestors exist and
    were simply not available to this catalog. Comparing a chain's head against
    this set is what separates that case from a product legitimately filed at a
    root, which the ``--require-category-path`` gate could not do while it only
    tested the path for emptiness.
    """
    return {n for n in taxonomy if not parents_of(taxonomy, n)}


def unanchored_head(
    chain: Sequence[str], taxonomy_roots: Set[str]
) -> Optional[str]:
    """The head of ``chain`` when the chain stops short of a global root.

    Returns ``None`` for a chain that is properly anchored — and for an empty
    chain, which is a different defect with its own counter (there is no
    resolved hierarchy at all to be truncated).
    """
    if not chain:
        return None
    head = chain[0]
    return None if head in taxonomy_roots else head


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

    ``None`` marks a root **of this map**, which is not the same thing as a root
    of the taxonomy: a node all of whose parents were pruned away by
    ``keep_prefixes`` or ``exclude`` also lands here parentless. Use
    :func:`global_roots` when the question is whether a chain reached the real
    top of the taxonomy.

    Build this **once per run** and thread it into :func:`category_chain` —
    rebuilding it per product would be both slow and pointless, since its whole
    purpose is to be product-independent.
    """
    nodes = eligible_nodes(taxonomy, keep_prefixes=keep_prefixes, exclude=exclude)

    parents: Dict[str, List[str]] = {
        n: parents_of(taxonomy, n, within=nodes) for n in nodes
    }
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
        label = display_label(taxonomy, node, lang)
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
    """Records where each category node landed and what it was called.

    Property 2 — one address per category — holds by construction once the chain
    walks a run-wide canonical parent map, but "by construction" is exactly the
    kind of claim that quietly stops being true. Feeding every emitted chain
    through here turns a regression into a line in the extraction report rather
    than a hand audit of the built index.

    The same argument applies to the node's *label*, in both directions, so this
    audits three things over every record written:

    * **one address per node** — a node's cumulative path is the same everywhere;
    * **one label per node** — the node reads identically in ``category_path``
      and in the flat ``categories`` list. Holds by construction now that both
      come from :func:`display_label`, and this is what proves it stayed true;
    * **one node per label** — the inverse. Two distinct nodes rendering the same
      string make the two fields ambiguous to join even with the labels unified,
      so it is a defect of the same join even though nothing is misspelled. It is
      zero among the nodes a path may contain — the taxonomy's 8,939
      English-backbone labels are all distinct, and so are the English/Spanish
      and English/French eligible sets — but *not* zero over the flat
      ``categories`` field, which carries the product's tags whatever their
      language prefix: upstream holds duplicate nodes such as
      ``en:capsicum-frutescens`` and ``fr:capsicum-frutescens`` that render to one
      string. A real English run over the first 200,000 export lines reports 2;
      French reports 6. The count is surfaced rather than suppressed because it
      bounds how far a label-to-segment join can be trusted.
    """

    def __init__(self) -> None:
        self.address: Dict[str, str] = {}
        self.conflicts: Dict[str, Set[str]] = {}
        self.label: Dict[str, str] = {}
        self.label_conflicts: Dict[str, Set[str]] = {}
        self.label_owners: Dict[str, Set[str]] = {}

    def record(
        self,
        path_entries: Iterable[Tuple[str, str]],
        flat_entries: Iterable[Tuple[str, str]] = (),
    ) -> None:
        """Take one product's ``(id, path)`` chain and its ``(id, label)`` tags.

        The chain's label for a node is the last segment of its cumulative path —
        :func:`display_label` neutralises the separator inside a label, so that
        split is exact and does not need the label passed in a second time.
        """
        for node, path in path_entries:
            seen = self.address.setdefault(node, path)
            if seen != path:
                self.conflicts.setdefault(node, {seen}).add(path)
            self._record_label(node, path.rsplit(PATH_SEPARATOR, 1)[-1])
        for node, label in flat_entries:
            self._record_label(node, label)

    def _record_label(self, node: str, label: str) -> None:
        seen = self.label.setdefault(node, label)
        if seen != label:
            self.label_conflicts.setdefault(node, {seen}).add(label)
        self.label_owners.setdefault(label, set()).add(node)

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)

    @property
    def label_conflict_count(self) -> int:
        """Nodes that rendered under more than one label. Must be zero."""
        return len(self.label_conflicts)

    @property
    def shared_label_count(self) -> int:
        """Labels claimed by more than one node — the join is ambiguous there."""
        return sum(1 for owners in self.label_owners.values() if len(owners) > 1)

    def summary(self, max_examples: int = 5) -> Dict[str, Any]:
        examples = [
            {"category": node, "addresses": sorted(paths)}
            for node, paths in sorted(self.conflicts.items())[:max_examples]
        ]
        label_examples = [
            {"category": node, "labels": sorted(labels)}
            for node, labels in sorted(self.label_conflicts.items())[:max_examples]
        ]
        shared = sorted(
            (label, owners)
            for label, owners in self.label_owners.items()
            if len(owners) > 1
        )[:max_examples]
        return {
            "categories_seen": len(self.address),
            "categories_at_multiple_addresses": self.conflict_count,
            "examples": examples,
            "labels_seen": len(self.label_owners),
            "categories_under_multiple_labels": self.label_conflict_count,
            "label_examples": label_examples,
            "labels_shared_by_multiple_categories": self.shared_label_count,
            "shared_label_examples": [
                {"label": label, "categories": sorted(owners)} for label, owners in shared
            ],
        }


class RootAnchorAudit:
    """Per-run totals for chains that stopped short of a global taxonomy root.

    ``--require-category-path`` refuses these records, and a refusal that is not
    counted is indistinguishable from an input that never contained one — the
    same argument that put the refused *tags* in the report. So the count and
    the offending heads are named per run whether or not the gate is on: with
    the gate off nothing is dropped and this block is the only place the number
    appears at all.

    The heads are worth naming rather than just totalling because the population
    is tiny and lopsided. Over the first 300,000 records of the January 2026
    dump, an English run resolves an unanchored chain for 455 products, and just
    six distinct heads account for all of them — two (``en:pate``,
    ``en:poultry-hams``) are 91% of the total. A list that short is a work item;
    a bare percentage is not.
    """

    def __init__(self, top_n: int = 20) -> None:
        self.top_n = top_n
        self.products = 0
        self.heads: Counter = Counter()

    def record(self, head: str) -> None:
        self.products += 1
        self.heads[head] += 1

    @property
    def distinct_heads(self) -> int:
        return len(self.heads)

    def summary(self) -> Dict[str, Any]:
        return {
            "products_with_unanchored_path": self.products,
            "distinct_unanchored_heads": self.distinct_heads,
            "top_unanchored_heads": [
                {"category": head, "products": n}
                for head, n in self.heads.most_common(self.top_n)
            ],
        }

    def log_lines(self, dropped: bool) -> List[str]:
        """One-screen summary for stderr; empty when nothing was unanchored."""
        if not self.products:
            return []
        verb = "dropped" if dropped else "kept (gate off)"
        lines = [
            f"Category path: {self.products:,} products {verb} on a chain that "
            f"stops short of a taxonomy root, under {self.distinct_heads:,} "
            "distinct heads. Their real ancestors are in the taxonomy but not in "
            "this catalog's languages.",
        ]
        top = ", ".join(
            f"{head} x{n:,}" for head, n in self.heads.most_common(5)
        )
        if top:
            lines.append(f"  top unanchored heads: {top}")
        return lines


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def open_text(path: Path) -> TextIO:  # small convenience used by tests
    return path.open("r", encoding="utf-8")
