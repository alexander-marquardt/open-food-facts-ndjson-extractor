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

To get a clean hierarchy in the shape retail catalogs typically expose (an array
of cumulative path strings like ``["Beverages", "Beverages/Hot beverages",
"Beverages/Hot beverages/Teas"]``) we need the taxonomy *graph* — the
parent→child edges — and then walk a single canonical chain from the product's
most specific category up to a root.

The taxonomy is the public OFF file:
    https://static.openfoodfacts.org/data/taxonomies/categories.json

Its shape is ``{canonical_id: {"name": {lang: label, ...},
"parents": [canonical_id, ...], "children": [...]}}``.

Strategy
--------
The address of a category must not depend on which product you are looking at.
So the graph is walked **once per run, globally**, over the whole taxonomy —
never per product:

1. :func:`build_canonical_parent_map` runs one BFS from the taxonomy's roots over
   the reversed ``parents`` edges. That gives every node its exact *fewest-hops*
   distance to a root, and picks each node's single canonical parent. It is built
   over the **whole** taxonomy, in every language, for every catalog.
2. :class:`AddressIndex` enumerates **every** root→node path over the full
   ``parents`` DAG. The path that follows the canonical parent at every hop is the
   node's **primary** address; the others are its **alternates**. Also built once
   per run, from the same global graph, so a node's whole address *set* is
   product-independent exactly as its single address used to be.
3. :func:`category_chain` takes the product's own tags only to choose the
   **leaf**, and walks the canonical parent map from that leaf all the way to a
   **global** root — materialising ancestors the product never tagged. That is the
   product's *primary* leaf; :func:`category_leaves` returns it followed by the
   alternates.
4. :func:`display_label` maps each canonical id to a display label (taxonomy
   ``name`` in the requested language, falling back to English, then ``xx``, then
   a prettified slug) and each address is emitted as cumulative ``/``-joined
   paths. That function is the single place a category's label is decided — the
   flat ``categories`` field calls it too, so the two fields can never disagree
   about what one node is called.

Several addresses per node, one of them primary
-----------------------------------------------
2,545 of the taxonomy's 14,457 nodes have more than one parent, so a node
genuinely sits at several addresses. Collapsing that to one made every product
reachable by exactly one breadcrumb, which is not what the source data says.
:func:`category_path_entries` therefore emits the **union** of the cumulative
entries across every address a product holds, and
:func:`primary_category_path_entries` emits the primary alone.

The canonical parent map is **not** deleted for this. It is retained and re-cast
as the *primary-address selector*, for three reasons:

* a product page shows one address plus "also categorized as …", so a primary has
  to exist regardless;
* the leaf-selection rules below keep their exact present meaning instead of
  needing redefinition — "longest canonical chain" and "drop canonical ancestors"
  are only ill-defined if the canonical map is gone;
* every path a run emitted before still exists, and every product's primary is
  byte-identical to what the collapsed build produced, so the rebuild diff is
  auditable: **a primary that moves is a defect, not noise.**

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
**Property 2 — one address *set* per category, and one primary address.** A
node's addresses come from the run-wide graph, so every occurrence of a node
resolves to the identical set of paths with the identical primary, on every
product. This is the restatement of the older "one address per category": what
was load-bearing about it was that the address does not depend on the product,
not that there is only one.

**Property 3 — a prefix-closed union of root→leaf chains per product.** A
product's ``category_path`` carries every cumulative prefix of every address it
holds, so a value's ancestor prefixes are always present as values of their own.
The older reading — exactly one chain — was the forest's shadow, not a property
of the source data.

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
  same set :func:`eligible_nodes` hands the flat ``categories`` field. The one
  prefix that is exempt is ``xx``, which marks a node named identically in every
  language and is therefore eligible in every catalog rather than in none; see
  :func:`default_keep_prefixes`.
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

import hashlib
import json
import sys
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
)

TAXONOMY_URL = "https://static.openfoodfacts.org/data/taxonomies/categories.json"

# The snapshot every catalog in ``builds/`` was made against, recorded there as
# ``pinned_taxonomy_sha256``. It is the *only* thing tying a build to a specific
# taxonomy: the 4.5MB file itself lives under ``data/``, which git ignores, so
# the digest travels in the repository and the bytes do not. A run that resolves
# some other file is not comparable to those builds, and this is the constant
# that lets the extractor say so instead of silently building anyway.
#
# Refreshing the taxonomy therefore means editing this line in a commit, which
# is the point: the address of every category is a function of this file, so a
# refresh moves `category_path` values and must be a reviewable act.
PINNED_TAXONOMY_SHA256 = "74717ecc001cf8661f6ec0bb3fc8c7a0cf317a6355a245004e892348fe575ec5"

# Path separator used *inside* each emitted cumulative path string. Matches the
# ``path_separator`` configured in the PRISM field map.
PATH_SEPARATOR = "/"

_HASH_CHUNK = 1 << 22


class TaxonomySnapshotError(RuntimeError):
    """The taxonomy snapshot on disk is not the one the run asked for.

    Raised instead of falling back to *some* taxonomy. A catalog built against
    an unintended snapshot is not detectably wrong afterwards — every product
    still gets a plausible ``category_path`` — so the only place to catch it is
    before the build starts.
    """


def taxonomy_sha256(path: Path) -> str:
    """Return the sha256 of ``path``, read in chunks (the snapshot is ~4.5MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_taxonomy(
    path: Path,
    *,
    fetch: bool = False,
    expected_sha256: Optional[str] = PINNED_TAXONOMY_SHA256,
    log: Optional[Any] = None,
) -> Path:
    """Return ``path`` once it is known to hold the snapshot the run asked for.

    Two refusals, both loud, replacing what used to be a silent download:

    * **The file is missing and ``fetch`` is false.** Downloading a fresh
      upstream taxonomy on a cache miss makes "the pin" mean "whatever upstream
      published today" exactly when a build needed it held fixed. Refreshing is
      now opt-in (``--fetch-taxonomy``), so it appears on the command line and
      therefore in the build record. The default path also sits inside
      ``data/json_source/``, a read-only source dump; a build has no business
      writing into it unasked.
    * **The file is not the expected snapshot.** ``path.exists()`` waves through
      a stale, truncated or hand-edited file. Pass ``expected_sha256=None`` to
      state deliberately that this run is not pinned.
    """
    if not path.exists():
        if not fetch:
            expected = expected_sha256 or "unpinned"
            raise TaxonomySnapshotError(
                f"category taxonomy snapshot not found at {path} "
                f"(expected sha256 {expected}). It is not downloaded automatically: "
                f"a build must not silently substitute today's upstream taxonomy for the "
                f"pinned one. Put the snapshot at that path, name another with --taxonomy, "
                f"or pass --fetch-taxonomy to download {TAXONOMY_URL} there on purpose."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if log:
            log(f"--fetch-taxonomy: downloading {TAXONOMY_URL} to {path} ...")
        try:
            with urllib.request.urlopen(TAXONOMY_URL) as resp:  # noqa: S310 (trusted OFF host)
                data = resp.read()
        except OSError as exc:  # URLError and friends are all OSError subclasses
            # An opt-in fetch that could not complete is still a run without its
            # taxonomy, and it exits through the same named error rather than a
            # traceback — the caller has one thing to catch, not two.
            raise TaxonomySnapshotError(
                f"could not download the category taxonomy from {TAXONOMY_URL} to {path}: {exc}"
            ) from exc
        path.write_bytes(data)
        if log:
            log(f"Saved category taxonomy ({len(data):,} bytes) to {path}")

    if expected_sha256:
        actual = taxonomy_sha256(path)
        if actual != expected_sha256:
            raise TaxonomySnapshotError(
                f"category taxonomy snapshot {path} is not the pinned one: "
                f"expected sha256 {expected_sha256}, found {actual}. "
                f"Every category address is a function of this file, so a build against it "
                f"would not be comparable to the pinned ones. Restore the pinned snapshot, "
                f"update PINNED_TAXONOMY_SHA256 in a commit if the pin is meant to move, "
                f"or pass --allow-unpinned-taxonomy to build against this file knowingly."
            )
        if log:
            log(f"Category taxonomy {path} matches the pinned sha256 {expected_sha256}.")
    elif log:
        log(f"Category taxonomy {path} used WITHOUT a sha256 pin (--allow-unpinned-taxonomy).")

    return path


def load_taxonomy(path: Path) -> Dict[str, Any]:
    """Load the OFF category taxonomy JSON into a ``{canonical_id: node}`` dict."""
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Taxonomy at {path} is not a JSON object")
    return obj


def _capitalise_first(label: str) -> str:
    """``label`` with its first character upper-cased, and **nothing else** touched.

    Not :meth:`str.capitalize`, which lower-cases everything after the first
    character and would destroy exactly what the taxonomy ``name`` is kept for:
    ``dried Toothed wrack`` would lose its species capital, ``farmed
    Mediterranean bass`` its proper noun. Only the first character is in
    question, and only when it is a lowercase letter — a name already starting
    with a capital, a digit or a punctuation mark comes back byte-identical,
    because ``str.islower`` is false for all of those. The snapshot has real
    names of each shape: ``10% red wine``, ``70% fat mayonnaise``, and the French
    ``% de matières grasses``.

    Why this is applied at all, when the rule everywhere else in this module is
    to render the upstream ``name`` verbatim: upstream does not capitalise
    consistently. 92 of the taxonomy's 8,939 ``en:``-prefixed nodes have an
    English name beginning lowercase — ``ice creams``, ``chorizo``, ``baker's
    yeast`` — so those segments read ``Desserts/Frozen desserts/Ice creams and
    sorbets/ice creams/Ice cream tubs`` mid-breadcrumb, measured on a real
    product, and since the flat tag field renders from the same labeller they
    read that way in a tag row too. 49 Spanish and 208 French names have the
    same defect.

    All 92 were enumerated against the pinned snapshot before this rule was
    chosen, because an unconditional upper-case is the wrong fix if any name is
    *deliberately* lowercase — a ``pH``-style term, a lowercase brand. None is:
    they are ordinary common nouns upstream simply did not capitalise, and no
    name in the snapshot, in any language, is of the ``pH``/``iPhone`` shape (a
    lowercase first character followed by an upper-case second one). They are
    recorded verbatim in ``tests/fixtures/off_real_lowercase_names.json`` so the
    next taxonomy refresh is checked against the same question rather than
    inheriting the answer.

    It is a presentation rule about the *first character of a label*, not an
    edit of the source: nothing here writes back to the taxonomy, and the
    ``name`` the label is taken from is unchanged in every other character.

    The slug fallback below already capitalised its first character, so before
    this the same node rendered ``Saint-émilion`` when the taxonomy had no name
    for it and ``saint-émilion`` when it did. One helper is now the single
    casing rule for both, which is why this is not a second transformation
    layered on the first.
    """
    return label[:1].upper() + label[1:] if label[:1].islower() else label


def _prettify_slug(canonical_id: str) -> str:
    """Fallback label for a canonical id with no usable taxonomy ``name``."""
    t = canonical_id
    if ":" in t:
        t = t.split(":", 1)[1]
    t = t.replace("-", " ").replace("_", " ").strip()
    if not t:
        return canonical_id
    return _capitalise_first(t)


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

    Whichever branch produced it, the label's **first character** is
    upper-cased if it is a lowercase letter, and no other character is touched —
    see :func:`_capitalise_first` for why upstream makes that necessary and why
    it is not :meth:`str.capitalize`. Both branches go through the same helper,
    so this is one casing rule rather than a second one layered on the name.
    """
    node = taxonomy.get(canonical_id) or {}
    names = node.get("name") if isinstance(node.get("name"), dict) else {}
    for key in (lang, "en", "xx"):
        val = names.get(key)
        if isinstance(val, str) and val.strip():
            label = _capitalise_first(val.strip())
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
    ``"en"`` — and ``"xx"``, which is not a language at all but upstream's marker
    for a node named identically everywhere; see
    :func:`default_keep_prefixes`, which is where that reasoning lives.

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
    """Pick each node's **primary** parent, once, for the whole run.

    The OFF category taxonomy is a DAG: 2,545 of its 14,457 nodes have more than
    one parent. This picks one of them per node, which is a spanning **forest** of
    that DAG — every non-root keeps exactly one primary parent, the roots stay
    roots, and no node is orphaned.

    **The other parent edges are no longer dropped.** They are enumerated by
    :class:`AddressIndex`, which treats the path following the choice made here at
    every hop as a node's *primary* address and every other root→node path as an
    *alternate*. This function is therefore the primary-address selector, not a
    flattening: keeping it is what lets every downstream rule below (leaf choice,
    ancestor coverage, the emitted breadcrumb) keep its exact present meaning
    while the alternates are added alongside.

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


class AddressExplosionError(RuntimeError):
    """A node has more root→node paths than any catalog could sensibly carry.

    Enumerating every path through a DAG is exponential in the worst case. The
    pinned snapshot's worst node has 28, so the cap this is raised at cannot fire
    on it — it exists so that a future taxonomy refresh which *is* pathological
    stops the build with a name instead of exhausting memory, or silently
    truncating a product's addresses, which would be a facet that lies.
    """


# Refuses at ~35x the pinned snapshot's worst node (28). Deliberately far above
# anything real, because the number is a circuit breaker and not a policy about
# how many addresses a category may have.
MAX_ADDRESSES_PER_NODE = 1000


class AddressIndex:
    """Every root→node address in the taxonomy DAG, primary first.

    Built **once per run** from the whole taxonomy, exactly like
    :func:`build_canonical_parent_map`, and for the same reason: a category's
    addresses must not depend on which product you are looking at. Nothing here
    consults a product, a language or a label — an address is a tuple of canonical
    ids, and :func:`display_label` renders it later.

    ``addresses(node)[0]`` is the **primary**: the path that follows
    ``canonical_parents`` at every hop, i.e. exactly what
    :func:`canonical_ancestry` returns and exactly what a run emitted before the
    alternates existed. The rest are the alternates, ordered shortest-first and
    then lexicographically — a total order over the *set* of paths, so it cannot
    move when the upstream file re-orders a ``parents`` list, which is the same
    stability argument the tie-break in the module docstring makes.

    Paths are built bottom-up in one pass over a topological order rather than by
    recursion: the taxonomy is 14,457 nodes deep enough to make a recursive walk a
    stack-depth question, and the iterative form also gives the cycle guard
    somewhere to stand.
    """

    def __init__(
        self,
        taxonomy: Dict[str, Any],
        canonical_parents: Dict[str, Optional[str]],
        exclude: Optional[Set[str]] = None,
        max_addresses_per_node: int = MAX_ADDRESSES_PER_NODE,
    ) -> None:
        nodes = eligible_nodes(taxonomy, keep_prefixes=None, exclude=exclude)
        self._parents: Dict[str, List[str]] = {
            n: sorted(set(parents_of(taxonomy, n, within=nodes))) for n in nodes
        }
        self._canonical = canonical_parents
        self._max = max_addresses_per_node
        self._addresses: Dict[str, Tuple[Tuple[str, ...], ...]] = {}
        self._build()

    def _build(self) -> None:
        # Iterative post-order: a node is resolved only once every parent is.
        # ``in_progress`` is the cycle guard — the pinned snapshot has none, but
        # ``build_canonical_parent_map`` already carries the same defence, and an
        # unguarded walk here would not terminate rather than merely be wrong.
        in_progress: Set[str] = set()
        for start in sorted(self._parents):
            if start in self._addresses:
                continue
            stack: List[str] = [start]
            while stack:
                node = stack[-1]
                if node in self._addresses:
                    stack.pop()
                    in_progress.discard(node)
                    continue
                pending = [
                    p
                    for p in self._parents[node]
                    if p not in self._addresses and p not in in_progress
                ]
                if pending:
                    in_progress.add(node)
                    stack.extend(pending)
                    continue
                stack.pop()
                in_progress.discard(node)
                self._addresses[node] = self._resolve(node)

    def _resolve(self, node: str) -> Tuple[Tuple[str, ...], ...]:
        paths: List[Tuple[str, ...]] = []
        for parent in self._parents[node]:
            # A parent still unresolved at this point is one the cycle guard cut;
            # its edge contributes nothing rather than looping forever.
            for prefix in self._addresses.get(parent, ()):
                if node in prefix:
                    continue  # a cycle edge would make the path revisit ``node``
                paths.append(prefix + (node,))
                if len(paths) > self._max:
                    raise AddressExplosionError(
                        f"category {node!r} has more than {self._max:,} root-to-node "
                        "addresses in this taxonomy. Every address becomes a "
                        "category_path value on every product filed beneath it, so "
                        "this would be a document, an index and a facet the catalog "
                        "cannot honestly serve. Refusing rather than truncating."
                    )
        if not paths:
            return ((node,),)
        distinct = set(paths)
        primary = tuple(canonical_ancestry(self._canonical, node))
        # ``primary`` is a path of this DAG by construction — every canonical
        # parent is a real parent — whenever the map was built over the same node
        # set. It is checked rather than assumed so that a caller who passes a
        # differently-scoped map gets a deterministic order instead of a leading
        # path the graph does not contain.
        if primary in distinct:
            rest = sorted(distinct - {primary}, key=lambda p: (len(p), p))
            return (primary,) + tuple(rest)
        return tuple(sorted(distinct, key=lambda p: (len(p), p)))

    def addresses(self, node: str) -> Tuple[Tuple[str, ...], ...]:
        """Every root→``node`` path, primary first. ``()`` for an unknown node."""
        return self._addresses.get(node, ())

    def primary(self, node: str) -> Tuple[str, ...]:
        """The primary root→``node`` path. ``()`` for an unknown node."""
        found = self._addresses.get(node)
        return found[0] if found else ()

    @property
    def multi_address_nodes(self) -> int:
        """Nodes sitting at more than one address — the DAG's forks, materialised."""
        return sum(1 for paths in self._addresses.values() if len(paths) > 1)

    @property
    def max_addresses(self) -> int:
        return max((len(paths) for paths in self._addresses.values()), default=0)

    def __len__(self) -> int:
        return len(self._addresses)


def category_chain(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    keep_prefixes: Optional[Set[str]] = None,
    canonical_parents: Optional[Dict[str, Optional[str]]] = None,
) -> List[str]:
    """Return this product's **primary** root→leaf chain of category ids.

    The product's own tags choose only the **leaf**; the rest of the chain comes
    from the run-wide canonical parent map and runs all the way to a **global**
    taxonomy root, materialising ancestors this product never tagged. That is
    what makes a node's primary address identical on every product carrying it.
    Returns ``[]`` when the product has no taxonomy-known category.

    **This function's result is frozen behaviour.** It is the primary breadcrumb,
    and the acceptance gate for restoring the DAG is that it did not move for a
    single product. The alternates live in :func:`category_leaves` and
    :class:`AddressIndex` alongside it, never inside it.

    Leaf selection — see :func:`category_leaves`, which is where the rule lives
    and which returns the alternates this discards.
    """
    if canonical_parents is None:
        canonical_parents = build_canonical_parent_map(taxonomy, exclude=exclude)
    leaves = category_leaves(
        product_tags,
        taxonomy,
        exclude,
        keep_prefixes=keep_prefixes,
        canonical_parents=canonical_parents,
    )
    return canonical_ancestry(canonical_parents, leaves[0]) if leaves else []


def category_leaves(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    keep_prefixes: Optional[Set[str]] = None,
    canonical_parents: Optional[Dict[str, Optional[str]]] = None,
) -> List[str]:
    """The nodes this product is filed under: **primary first**, then alternates.

    A product's tags can sit on several disjoint branches, and only one of them
    can head the breadcrumb the product page leads with. The rule that picks that
    one is unchanged, and picks the same node it always did:

    0. Keep only tags whose language prefix is in ``keep_prefixes``. This is the
       **one** place the catalog's language narrows the hierarchy: an English
       product is not filed under a French-only category. It is applied to the
       product's tags, not to the graph, so a leaf chosen here still walks
       through whatever ancestors the taxonomy gives it. ``None`` accepts any
       language.
    1. Drop any tag that is a canonical ancestor of another of this product's
       tags — an ancestor is already on the more specific tag's chain.
    2. Of what remains, the **longest** canonical chain wins (the most specific
       filing), and **on a tie the lexicographically smallest canonical id
       wins** — the same tie-break the parent map uses.

    What changed is only that step 2's losers are **returned rather than
    discarded**. They were a real second filing of the product all along; dropping
    them is the second of the two collapses that made every product reachable by
    exactly one breadcrumb, and relaxing the graph alone would not have removed it.
    The order is the same total order step 2 minimises over, so the alternates are
    as stable as the primary.

    Coverage in step 1 is still computed over **canonical** chains, deliberately.
    Widening it to the whole DAG would change which node is primary, and the
    primary is the thing this restoration must not move. An alternate leaf that
    does turn out to be a DAG-ancestor of another costs nothing: its addresses are
    prefixes of that other leaf's, so the union de-duplicates them away.

    ``canonical_parents`` should be the map built once per run by
    :func:`build_canonical_parent_map`. It is optional only so ad-hoc callers and
    tests can pass tags alone; omitting it rebuilds the map on every call, which
    is far too slow for an extraction run.
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

    return sorted(candidates, key=lambda n: (-len(chains[n]), n))


class CategoryAddresses(NamedTuple):
    """One product's category addressing, primary and union in one pass.

    ``primary`` and ``entries`` are produced together because they share the leaf
    selection and the graph walk, and because a caller that takes one without the
    other is almost always the beginning of a bug: the emitted field is the union
    and the breadcrumb is the primary, and the two must describe the same product.
    """

    #: ``(canonical_id, cumulative_path)`` along the **primary** address only.
    #: Byte-identical to what a run emitted before alternates existed.
    primary: List[Tuple[str, str]]
    #: ``(canonical_id, cumulative_path)`` over **every** address, primary first.
    #: De-duplicated on the *pair*: one id may legitimately appear at several
    #: paths, and — see :func:`category_path_entries` — one path may legitimately
    #: be claimed by several ids.
    entries: List[Tuple[str, str]]


def category_addresses(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    lang: str = "en",
    canonical_parents: Optional[Dict[str, Optional[str]]] = None,
    address_index: Optional[AddressIndex] = None,
) -> CategoryAddresses:
    """Every address this product sits at, with its primary named separately.

    The addresses are, in order: every address of the primary leaf (its own
    primary first), then every address of each alternate leaf. So
    ``entries[:len(primary)] == primary`` and the breadcrumb a product page leads
    with is the head of the same list the facet counts over.

    ``address_index`` should be the :class:`AddressIndex` built once per run.
    Omitting it rebuilds the whole DAG enumeration on every call, which is far too
    slow for an extraction run and is offered only so ad-hoc callers and tests can
    pass tags alone.
    """
    if canonical_parents is None:
        canonical_parents = build_canonical_parent_map(taxonomy, exclude=exclude)
    if address_index is None:
        address_index = AddressIndex(taxonomy, canonical_parents, exclude=exclude)

    leaves = category_leaves(
        product_tags,
        taxonomy,
        exclude,
        keep_prefixes=default_keep_prefixes(lang),
        canonical_parents=canonical_parents,
    )

    primary: List[Tuple[str, str]] = []
    entries: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    for index, leaf in enumerate(leaves):
        for address in address_index.addresses(leaf):
            rendered = _render_address(taxonomy, address, lang)
            if index == 0 and not primary:
                primary = rendered
            for pair in rendered:
                if pair not in seen:
                    seen.add(pair)
                    entries.append(pair)
    return CategoryAddresses(primary=primary, entries=entries)


def _render_address(
    taxonomy: Dict[str, Any], address: Sequence[str], lang: str
) -> List[Tuple[str, str]]:
    """``(canonical_id, cumulative_path)`` for each step of one root→leaf address."""
    rendered: List[Tuple[str, str]] = []
    prefix = ""
    for node in address:
        label = display_label(taxonomy, node, lang)
        prefix = label if not prefix else f"{prefix}{PATH_SEPARATOR}{label}"
        rendered.append((node, prefix))
    return rendered


def category_path_entries(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    lang: str = "en",
    canonical_parents: Optional[Dict[str, Optional[str]]] = None,
    address_index: Optional[AddressIndex] = None,
) -> List[Tuple[str, str]]:
    """``(canonical_id, cumulative_path)`` over **every** address of this product.

    Callers that only need the paths want :func:`build_category_path`. **The id is
    not decoration and must not be re-derived from the path string**: the two are
    not in bijection, and were not even before the alternates existed. On the
    pinned snapshot four nodes render to a full path string another node also
    claims — one of them live, where a French catalog's
    ``…/Vins italiens/Chianti`` is claimed by both ``en:chianti`` and
    ``it:chianti``. Any consumer going breadcrumb → category id has to read the
    pair.

    A node may now appear more than once, once per address it holds; the list is
    de-duplicated on the pair, not on either half of it. For the single address a
    product leads with, see :func:`primary_category_path_entries`.
    """
    return category_addresses(
        product_tags,
        taxonomy,
        exclude,
        lang,
        canonical_parents=canonical_parents,
        address_index=address_index,
    ).entries


def primary_category_path_entries(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    lang: str = "en",
    canonical_parents: Optional[Dict[str, Optional[str]]] = None,
    address_index: Optional[AddressIndex] = None,
) -> List[Tuple[str, str]]:
    """``(canonical_id, cumulative_path)`` along the product's **primary** address.

    The breadcrumb a product page leads with, and a prefix of
    :func:`category_path_entries`. Frozen behaviour: this is the list the
    DAG-restoration acceptance gate compares against the pre-restoration build,
    product by product.
    """
    return category_addresses(
        product_tags,
        taxonomy,
        exclude,
        lang,
        canonical_parents=canonical_parents,
        address_index=address_index,
    ).primary


def default_keep_prefixes(lang: str) -> Set[str]:
    """Taxonomy languages a catalog in ``lang`` may **name** a category in.

    English is the taxonomy backbone; the persona language rides alongside it so
    localized-only nodes are still filable while foreign-language noise stays out
    of the leaf and out of the flat ``categories`` field.

    ``xx`` is always in the set, and is not a language. Upstream uses that prefix
    for a node whose name is *the same in every language* — 34 of them in the
    pinned snapshot, ``xx:tofu``, ``xx:dumplings``, ``xx:sake`` and the
    like. Refusing it is not the deliberate refusal ``fr:pates-a-tartiner`` gets
    from an English catalog: a French-only node genuinely has no business being a
    searchable English category, whereas a language-neutral node belongs to
    **every** catalog by construction, and refusing it from all of them left 34
    real categories that no catalog could ever emit. The inconsistency was
    already visible one function away, in :func:`display_label`, which reads an
    ``xx`` *name* happily — so ``xx`` was accepted as a legitimate way to name a
    node while being refused as a way to *be* one.

    It does not bound what a chain may *pass through*: a path's intermediate
    nodes come from the language-blind graph and are localized by
    :func:`display_label` instead. Applying this to the graph is what created 90
    phantom roots for an English run.
    """
    return {"en", "xx", lang} if lang else {"en", "xx"}


def path_strings(entries: Iterable[Tuple[str, str]]) -> List[str]:
    """The distinct cumulative paths of ``entries``, in first-seen order.

    De-duplicated on the **path**, not on the pair: two different nodes can render
    to one string (see :func:`category_path_entries`), and the emitted field is a
    set of addresses, so it must carry that string once. The ids are what
    distinguish them, and they stay in ``entries``.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for _node, path in entries:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def build_category_path(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    lang: str = "en",
    canonical_parents: Optional[Dict[str, Optional[str]]] = None,
    address_index: Optional[AddressIndex] = None,
) -> List[str]:
    """Cumulative path strings over **every** address this product sits at.

    Example, for a product on one address: ``["Beverages",
    "Beverages/Hot beverages", "Beverages/Hot beverages/Teas",
    "Beverages/Hot beverages/Teas/Tea bags"]``. A product on several addresses
    carries the union of their cumulative entries, the primary address first — the
    shape that makes it reachable by more than one breadcrumb and that a facet
    aggregation counts over.

    For the single address the product page leads with, see
    :func:`build_primary_category_path`.
    """
    return path_strings(
        category_path_entries(
            product_tags,
            taxonomy,
            exclude,
            lang,
            canonical_parents=canonical_parents,
            address_index=address_index,
        )
    )


def build_primary_category_path(
    product_tags: List[str],
    taxonomy: Dict[str, Any],
    exclude: Set[str],
    lang: str = "en",
    canonical_parents: Optional[Dict[str, Optional[str]]] = None,
    address_index: Optional[AddressIndex] = None,
) -> List[str]:
    """Cumulative path strings along the product's **primary** address alone."""
    return path_strings(
        primary_category_path_entries(
            product_tags,
            taxonomy,
            exclude,
            lang,
            canonical_parents=canonical_parents,
            address_index=address_index,
        )
    )


class AddressAudit:
    """Records where each category node landed and what it was called.

    Property 2 — one *primary* address per category — holds by construction once
    the primary walks a run-wide canonical parent map, but "by construction" is
    exactly the kind of claim that quietly stops being true. Feeding every emitted
    address through here turns a regression into a line in the extraction report
    rather than a hand audit of the built index.

    **What the restored DAG changed here, and what it did not.** A node now sits at
    several addresses on purpose, so "a node's path is the same everywhere" is no
    longer the invariant — but the thing that was load-bearing about it survives
    verbatim: a node's addressing must not depend on which product you are looking
    at. So the conflict counter watches the **primary** address, where a
    disagreement is still a defect and the expected value is still zero, and the
    alternates are counted separately as the shape they are.

    The same argument applies to the node's *label*, in both directions, so this
    audits three things over every record written:

    * **one primary address per node** — a node's primary cumulative path is the
      same everywhere;
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
        self.addresses: Dict[str, Set[str]] = {}
        self.products = 0
        self.multi_address_products = 0
        self.path_values = 0
        self.max_path_values = 0
        self.distinct_paths: Set[str] = set()

    def record(
        self,
        path_entries: Iterable[Tuple[str, str]],
        flat_entries: Iterable[Tuple[str, str]] = (),
        all_entries: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> None:
        """Take one product's primary ``(id, path)`` chain and its ``(id, label)`` tags.

        ``all_entries`` is the same product's entries over **every** address it
        holds — what actually reaches the ``category_path`` field. It defaults to
        ``path_entries``, so a caller that emits only a primary is audited exactly
        as before rather than reporting a vacuous zero for the alternates.

        The chain's label for a node is the last segment of its cumulative path —
        :func:`display_label` neutralises the separator inside a label, so that
        split is exact and does not need the label passed in a second time.
        """
        primary = list(path_entries)
        emitted = primary if all_entries is None else list(all_entries)

        for node, path in primary:
            seen = self.address.setdefault(node, path)
            if seen != path:
                self.conflicts.setdefault(node, {seen}).add(path)
        for node, path in emitted:
            self.addresses.setdefault(node, set()).add(path)
            self._record_label(node, path.rsplit(PATH_SEPARATOR, 1)[-1])
        for node, label in flat_entries:
            self._record_label(node, label)

        self.products += 1
        values = path_strings(emitted)
        self.path_values += len(values)
        self.max_path_values = max(self.max_path_values, len(values))
        self.distinct_paths.update(values)
        # A product is multi-address when the union carries strictly more than its
        # primary — counted from the emitted values rather than from the leaf
        # count, because two thirds of these products have a single leaf and fork
        # *above* it, and a leaf-based count undercounts them by about 2.5x.
        if len(values) > len(path_strings(primary)):
            self.multi_address_products += 1

    def _record_label(self, node: str, label: str) -> None:
        seen = self.label.setdefault(node, label)
        if seen != label:
            self.label_conflicts.setdefault(node, {seen}).add(label)
        self.label_owners.setdefault(label, set()).add(node)

    @property
    def conflict_count(self) -> int:
        return len(self.conflicts)

    @property
    def multi_address_category_count(self) -> int:
        """Categories emitted at more than one address. A shape, not a defect."""
        return sum(1 for paths in self.addresses.values() if len(paths) > 1)

    @property
    def max_addresses_for_a_category(self) -> int:
        return max((len(paths) for paths in self.addresses.values()), default=0)

    @property
    def mean_path_values(self) -> float:
        """Mean ``category_path`` values per record written.

        Reported because it is the number that sizes the index: it is the postings
        per document on the field, and ``distinct_category_paths`` beside it is
        the bucket count every ``category_path`` terms aggregation downstream has
        to be sized against. An aggregation still sized for the pre-restoration
        cardinality truncates silently, and a truncated facet panel lies.
        """
        return self.path_values / self.products if self.products else 0.0

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
        multi = sorted(
            (node, paths)
            for node, paths in self.addresses.items()
            if len(paths) > 1
        )[:max_examples]
        return {
            "categories_seen": len(self.address),
            "categories_at_multiple_primary_addresses": self.conflict_count,
            "examples": examples,
            "products": self.products,
            "products_at_multiple_addresses": self.multi_address_products,
            "categories_with_alternate_addresses": self.multi_address_category_count,
            "max_addresses_for_a_category": self.max_addresses_for_a_category,
            "multi_address_examples": [
                {"category": node, "addresses": sorted(paths)} for node, paths in multi
            ],
            "distinct_category_paths": len(self.distinct_paths),
            "mean_category_path_values": round(self.mean_path_values, 3),
            "max_category_path_values": self.max_path_values,
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
