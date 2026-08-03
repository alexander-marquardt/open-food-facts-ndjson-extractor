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
   node is orphaned, only redundant parent edges are dropped. It is built over
   the **whole** taxonomy, in every language, for every catalog.
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

Where the language filter applies — naming, not traversal
----------------------------------------------------------
A catalog is localized, so it must not offer ``fr:pates-a-tartiner`` as an
English category. But pruning the *graph* to enforce that also deleted every
edge through a pruned node, so a node whose only parent was foreign became a
root of the pruned graph and its chain stopped there. Measured on the pinned
snapshot, that promoted 90 nodes to roots for an English run (161 roots against
the taxonomy's 92), 90 for Spanish and 53 for French — three catalogs each
projecting against a differently shaped forest. ``en:pate`` sits under
``fr:charcuteries-diverses`` under ``en:prepared-meats``, and an English catalog
filed it as a top-level ``Pâté``.

So the language filter is applied where the localization argument actually
holds, and nowhere else:

* **Traversal is language-blind.** :func:`build_canonical_parent_map` covers all
  14,457 nodes for every catalog, so its roots *are* the taxonomy's 92 and a
  chain can pass through a foreign ancestor instead of being severed at it. One
  map serves en, es and fr, so a node's ancestry is the same list of ids in all
  three (92 nodes disagreed before).
* **Leaf choice is language-filtered.** ``keep_prefixes`` still decides which of
  the product's *own* tags may be the leaf, which is what keeps a French-only
  category from being the thing an English product is filed under, and it is the
  same set :func:`eligible_nodes` hands the flat ``categories`` field.
* **Labels are localized, by fallback.** A traversed foreign node renders
  through :func:`display_label` like any other — ``lang`` → English → ``xx`` →
  slug — so the ``fr:`` hop in ``en:pate``'s lineage reads ``Charcuteries
  diverses`` in the English path. Measured: of the 8,939 nodes an English
  catalog may file under, 141 have a foreign node in their global ancestry and
  81 have one with no English or ``xx`` name, so 0.9% of English filings carry
  one French-language segment. That is the price of the trade, and it buys back
  the true lineage of every one of them.

What that leaves for the ``--require-category-path`` gate is ``--category-
exclude``: an operator who excludes a mid-taxonomy node still severs the chains
beneath it, and those chains are still truncated addresses that no other catalog
agrees with. :func:`global_roots` and :func:`unanchored_head` therefore stay, and
the gate still consults them rather than only testing the path for emptiness —
the number it reports is simply expected to be zero now.

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
    so on the English backbone the slug fallback never fires. It fires for the
    foreign ancestors a language-blind chain now walks through: of the 8,965
    nodes an English path may contain, 26 are foreign and 20 of those have
    neither an English nor an ``xx`` name, so they render de-slugged from their
    own language — ``fr:charcuteries-diverses`` reads ``Charcuteries diverses``.
    That is the deliberate trade for keeping their descendants' real lineage; see
    the module docstring. It also remains the fallback for runs with no taxonomy
    at all (``--no-taxonomy``, which passes an empty mapping here).
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

    ``keep_prefixes`` (e.g. ``{"en"}``) keeps an English catalog from *naming* a
    French-only node like ``fr:pates-a-tartiner``: it decides which ids may be a
    product's leaf and which may be a value of the flat ``categories`` field.
    English is the taxonomy's backbone language, so callers typically include
    ``"en"``.

    It deliberately does **not** decide which nodes a chain may walk *through*.
    Filtering the graph deleted the edges through every filtered node too, which
    orphaned their children into roots of a locale-shaped forest; see the module
    docstring. ``None`` — no filtering at all — is what the traversal graph asks
    for.
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
    """Taxonomy ids this catalog is allowed to **name**.

    That is: the ids a product may be filed *under* (the leaf of its chain) and
    the ids the flat ``categories`` field may carry. Both fields draw on this one
    set so they can never disagree about what exists.

    This is **not** the set a chain may walk through — see
    :func:`build_canonical_parent_map`, which is language-blind on purpose. Pass
    ``keep_prefixes=None`` to ask for the whole taxonomy minus ``exclude``.
    """
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
    which is the whole point. The language filter no longer touches the graph, so
    the 92 roots here are exactly the roots of the map
    :func:`build_canonical_parent_map` builds for a default run — that agreement
    is the invariant, not a coincidence, and the gate is what proves it held.

    ``--category-exclude`` can still break it: excluding a mid-taxonomy node
    strands every node beneath it, which becomes a root **of the map** while
    remaining a child in the taxonomy. A chain headed by such a node is
    *truncated* — its real ancestors exist and were simply not available to this
    run. Comparing a chain's head against this set is what separates that case
    from a product legitimately filed at a root, which the
    ``--require-category-path`` gate could not do while it only tested the path
    for emptiness.
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
    exclude: Optional[Set[str]] = None,
) -> Dict[str, Optional[str]]:
    """Pick one canonical parent for every taxonomy node, once, for the whole run.

    The OFF category taxonomy is a DAG: 2,545 of its 14,457 nodes have more than
    one parent. This collapses it to a spanning **forest** — every non-root keeps
    exactly one parent, the roots stay roots, and no node is orphaned; only
    redundant parent edges are dropped.

    **Language-blind, and takes no ``keep_prefixes``.** The graph is the same for
    every catalog: all 14,457 nodes, so the map's roots are the taxonomy's 92 and
    a chain crosses a foreign ancestor rather than stopping at it. The parameter
    used to exist and is gone rather than defaulted, because a pruned graph is
    not a weaker version of this one — it is a differently shaped forest with
    90 extra roots for an English run, and no caller should be able to ask for it
    by accident. Where language *does* apply is :func:`eligible_nodes`.

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

    ``None`` marks a root **of this map**. With no ``exclude`` those are exactly
    :func:`global_roots`; ``exclude`` can still strand a node whose only parents
    were excluded, so that function stays the authority on whether a chain
    reached the real top of the taxonomy.

    Build this **once per run** and thread it into :func:`category_chain` —
    rebuilding it per product would be both slow and pointless, since its whole
    purpose is to be product-independent.
    """
    nodes = eligible_nodes(taxonomy, keep_prefixes=None, exclude=exclude)

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

    0. Keep only tags whose language prefix is in ``keep_prefixes``. This is the
       **one** place the catalog's language narrows the hierarchy: an English
       product is not filed under a French-only category. It is applied to the
       product's tags, not to the graph, so a chain chosen here still walks
       through whatever ancestors the taxonomy gives it. ``None`` accepts any
       language.
    1. Drop any tag that is a canonical ancestor of another of this product's
       tags — an ancestor is already on the more specific tag's chain.
    2. Of what remains, the **longest** canonical chain wins (the most specific
       filing), and **on a tie the lexicographically smallest canonical id
       wins** — the same tie-break the parent map uses.

    ``canonical_parents`` should be the map built once per run by
    :func:`build_canonical_parent_map`. It is optional only so ad-hoc callers and
    tests can pass tags alone; omitting it rebuilds the map on every call, which
    is far too slow for an extraction run. Supplying it does not change the
    outcome: the map is language-blind either way, and ``keep_prefixes`` is
    consulted here regardless of which branch produced it.
    """
    if canonical_parents is None:
        canonical_parents = build_canonical_parent_map(taxonomy, exclude=exclude)

    leaf_ok = _lang_filter(keep_prefixes)
    present: Set[str] = {
        t
        for t in product_tags
        if t not in exclude and t in canonical_parents and leaf_ok(t)
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
    """Taxonomy languages a catalog in ``lang`` may **name** a category in.

    English is the taxonomy backbone; the persona language rides alongside it so
    localized-only nodes are still filable while foreign-language noise stays out
    of the leaf and out of the flat ``categories`` field.

    It does not bound what a chain may *pass through*: a path's intermediate
    nodes come from the language-blind graph and are localized by
    :func:`display_label` instead. Applying this to the graph is what created 90
    phantom roots for an English run.
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
      zero among the nodes a path may contain — re-measured once traversal went
      language-blind, over the 8,965 / 9,354 / 11,753 nodes reachable in an
      English, Spanish and French path, all of whose labels are distinct — but
      *not* zero over the flat
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

    **On a default run the expected value is zero**, and that is the reason to
    keep reporting it rather than to stop. The language filter used to sever
    chains wholesale — over the first 300,000 records of the January 2026 dump an
    English run left 455 products unanchored under six heads, ``en:pate`` and
    ``en:poultry-hams`` alone being 91% of them — and now does not, because
    traversal is language-blind. What can still sever a chain is
    ``--category-exclude`` naming a mid-taxonomy node, which is an operator
    decision whose cost belongs in the report. A zero here is the standing
    evidence that the forest still has exactly the taxonomy's roots; a non-zero
    one names the excluded node's orphans.
    """

    def __init__(
        self,
        taxonomy_roots: Optional[Set[str]] = None,
        traversal_roots: Optional[Set[str]] = None,
        top_n: int = 20,
    ) -> None:
        self.top_n = top_n
        self.products = 0
        self.heads: Counter = Counter()
        # The two forests, so the report answers "did pruning invent roots?"
        # without anyone re-deriving it. That question is the *cause* of every
        # product this audit can count, and it is answerable before a single
        # record is read — a run whose input happens to carry nothing under a
        # phantom root would otherwise report a clean zero over a broken forest.
        self.taxonomy_roots: Set[str] = set(taxonomy_roots or ())
        self.traversal_roots: Set[str] = set(traversal_roots or ())

    def record(self, head: str) -> None:
        self.products += 1
        self.heads[head] += 1

    @property
    def distinct_heads(self) -> int:
        return len(self.heads)

    @property
    def phantom_roots(self) -> List[str]:
        """Roots of the walked graph that the taxonomy does not call roots."""
        return sorted(self.traversal_roots - self.taxonomy_roots)

    def summary(self) -> Dict[str, Any]:
        phantom = self.phantom_roots
        return {
            "taxonomy_roots": len(self.taxonomy_roots),
            "traversal_roots": len(self.traversal_roots),
            "phantom_roots": len(phantom),
            "top_phantom_roots": phantom[: self.top_n],
            "products_with_unanchored_path": self.products,
            "distinct_unanchored_heads": self.distinct_heads,
            "top_unanchored_heads": [
                {"category": head, "products": n}
                for head, n in self.heads.most_common(self.top_n)
            ],
        }

    def log_lines(self, dropped: bool) -> List[str]:
        """One-screen summary for stderr; empty when the forest and run are clean."""
        lines: List[str] = []
        phantom = self.phantom_roots
        if phantom:
            # Louder than the per-product count, and printed even when no product
            # happened to sit under one: a graph with roots the taxonomy does not
            # have is a defect of the whole run's addressing, not of those rows.
            lines.append(
                f"WARNING: the category graph this run walked has {len(phantom):,} "
                f"roots the taxonomy does not ({len(self.traversal_roots):,} against "
                f"{len(self.taxonomy_roots):,}). Every chain under one of them is "
                "filed at an address no other catalog agrees with."
            )
            lines.append(f"  phantom roots: {', '.join(phantom[:5])}")
        if not self.products:
            return lines
        verb = "dropped" if dropped else "kept (gate off)"
        lines += [
            f"Category path: {self.products:,} products {verb} on a chain that "
            f"stops short of a taxonomy root, under {self.distinct_heads:,} "
            "distinct heads. Their real ancestors are in the taxonomy but were "
            "excluded from this run, so the chain has nowhere to anchor.",
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
