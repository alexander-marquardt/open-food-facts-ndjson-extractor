"""Curation of a product's ``categories_tags`` before anything is indexed.

The problem
-----------
Open Food Facts ships ``categories_tags`` on every product, and those tags are
**not** guaranteed to be nodes of the category taxonomy. Legacy tags, tags
entered by contributors, and tags that upstream has since renamed all ride along
in the export. Measured over the first 300,000 records of the January 2026 dump
against the pinned 14,457-node snapshot:

===============================  =========  =======
products with a real tag           218,239
tag instances                    1,171,851
instances with no taxonomy node     49,962   (4.26%)
distinct such tags                   7,712
products carrying at least one      42,727   (19.6%)
products where *every* tag is one     5,456   (2.50%)
===============================  =========  =======

Until this module existed those tags were prettified straight into the indexed,
searchable ``categories`` field as if they were canonical. ``Groceries`` alone
was searchable on 6,299 documents of the built English catalog — a value that
has no place in the hierarchy, can never be a ``category_path`` segment, and
cannot be authored as a policy value.

Values are dropped, records are not
-----------------------------------
Dropping the whole record on an unresolvable tag would cost 19.6% of tagged
products; dropping only the offending value costs 2.5%. In 87% of affected cases
the product carries a complete valid lineage *and* one junk tag riding along —
``en:groceries`` is on 18,291 tag instances but only 58 of its carriers have
nothing else. So a rejected tag never rejects its product. The 2.5% that end up
with no category at all are exactly the products the existing
``--require-category-path`` gate already drops.

The three rejection reasons
---------------------------
``curated_drop``
    A tag on :data:`TAG_DROPS`: it is contentless, or it is an *attribute*
    rather than a category. Recorded with its reason so the decision is legible
    instead of buried in the long tail.
``not_in_taxonomy``
    No node with this id, in the pinned snapshot or upstream. The long tail.
``out_of_language``
    A real taxonomy node, but in a language this catalog does not file products
    under (see ``taxonomy.default_keep_prefixes``). ``fr:charcuteries-cuites`` is
    a genuine node and still has no business being a searchable English
    category: ``category_chain`` will not choose it as a leaf either, so leaving
    it in the flat field produces a value that can be searched but never faceted
    or filtered — the same defect ``en:groceries`` had. It costs 3 further
    products out of 218,239 to refuse it (measured), because a product with a
    French-only tag almost always carries the English lineage too.

    Refused as a *value*, not erased from the hierarchy: the same node may still
    appear as an intermediate ``category_path`` segment, because a path walks the
    language-blind taxonomy graph. An English product tagged
    ``fr:charcuteries-cuites`` is not filed under it, but a product filed under
    ``en:poultry-hams`` does pass through it, which is where that node genuinely
    sits.

Aliases are canonicalisations, never generalisations
----------------------------------------------------
:data:`TAG_ALIASES` maps a retired id to the node that *is the same category*
under a new id. It deliberately does **not** map a tag to a broader node: filing
``en:raw-cured-ham`` under ``en:cured-hams`` would put a product in a category it
never claimed, which is a worse failure than dropping the value — and the
measurement says nothing is at stake, since those carriers keep a valid lineage
anyway.

Each entry was verified, not guessed. Two evidence classes:

**A — upstream synonym line.** When upstream renames a category it keeps the old
name as a synonym on the surviving node's language line in the taxonomy source
(``taxonomies/food/categories.txt`` in ``openfoodfacts/openfoodfacts-server``).
Slugifying every synonym of every entry yields an authoritative retired-id →
canonical-id index; the class-A entries below are that index, restricted to the
ids the dump actually carries. ``en: Easter foods and drinks, Easter food`` is
the shape of the evidence.

**B — name equivalence.** No surviving synonym, but the successor is
unmistakable and the mapping rescues products that would otherwise have no
category at all. Two entries, both marked inline.

Every alias target was checked to exist in the pinned snapshot.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .taxonomy import eligible_nodes

# Rejection reason codes. Reported per run so a regression is a moving number in
# the report rather than something rediscovered by hand against the built index.
REASON_CURATED_DROP = "curated_drop"
REASON_NOT_IN_TAXONOMY = "not_in_taxonomy"
REASON_OUT_OF_LANGUAGE = "out_of_language"


# Retired id -> the node that is the same category today. Counts are instances in
# the first 300,000 records of the January 2026 dump. Class A unless marked.
TAG_ALIASES: Dict[str, str] = {
    # -- B: no surviving upstream synonym; "salted" and "salty" name one category,
    # and every child of the successor is a `en:salted-*` node. Worth the most of
    # any entry here: 2,387 of its 2,998 carriers have no other usable tag, so
    # without it they are dropped by the category_path gate.
    "en:salted-snacks": "en:salty-snacks",  # 2998 Salty snacks
    # -- B: the taxonomy has exactly one decoration node and this is the same
    # product class (sprinkles, icing, cake toppers). 991 of 1,098 carriers have
    # no other usable tag.
    "en:baking-decorations": "en:food-decorations",  # 1098 Food decorations
    # -- A: retired ids kept as synonyms on the successor's language line.
    "en:squeezed-juices": "en:squeezed-fruit-juices",  # 540
    "en:easter-food": "en:easter-foods-and-drinks",  # 224
    "en:melted-cheese": "en:processed-cheeses",  # 192
    "en:gherkins": "en:pickled-gherkins",  # 181
    "en:processed-cheese": "en:processed-cheeses",  # 170
    "en:drinkable-yogurts": "en:yogurt-drinks",  # 151
    "fr:cremes-fraiches": "en:cremes-fraiches",  # 150
    "en:puffed-salty-snacks-potato-soy-based": "en:puffed-salty-snacks-made-from-potato-and-soy",  # 144
    "en:tuna-in-brine": "en:tunas-in-brine",  # 142
    "en:dried-mixed-fruits": "en:mixed-dried-fruits",  # 129
    "en:bars-covered-with-chocolate": "en:candy-chocolate-bars",  # 123
    "en:fries": "en:potato-fries",  # 108
    "en:cane-sugar": "en:cane-sugars",  # 104
    "en:tofu": "xx:tofu",  # 101
    "en:chamomile": "en:camomile-teas",  # 93
    "en:mandarin-oranges": "en:mandarins",  # 92
    "en:king-cakes": "en:puff-pastry-king-cakes",  # 84
    "en:frozen-fries": "en:frozen-potato-fries",  # 81
    "en:faiselles": "en:faisselles",  # 78
    "en:roasted-salted-almonds": "en:roasted-and-salted-almonds",  # 75
    "en:chocolate-powders": "en:instant-chocolate-powders",  # 70
    "en:bilberries-jams": "en:bilberry-jams",  # 64
    "en:oatmeal-cookies": "en:oatmeal-biscuits",  # 63
    "en:chocolate-ice-cream-bars": "en:ice-cream-bars-coated-with-chocolate",  # 56
    "fr:poulets-fermiers": "en:chicken-free-range-meat-and-skin-raw",  # 55
    "en:chocolate-rabbits": "en:easter-chocolate-rabbits",  # 53
    "af:fusilli": "en:fusilli",  # 50
}


# Tags refused on purpose, each with the reason it is refused. None of these is a
# node of the pinned snapshot *or* of the current upstream taxonomy — they are
# listed anyway so the decision is recorded and reported separately, instead of
# vanishing into a 7,712-entry tail where nobody can see that one tag is a third
# of the problem.
TAG_DROPS: Dict[str, str] = {
    "en:groceries": (
        "contentless catch-all — every product in a grocery catalog is groceries. "
        "36.6% of all unresolvable tag instances, and only 58 of its 18,291 "
        "carriers have no other tag, so refusing it costs essentially nothing."
    ),
    "fr:autres-produits": (
        "contentless catch-all ('other products') — names the absence of a "
        "category rather than a category."
    ),
    "en:aoc-cheeses": (
        "protected-designation label, not a category. The designation is already "
        "carried on the product as a label; its carriers all keep their cheese "
        "lineage, so nothing is lost by keeping it out of the category vocabulary."
    ),
    "fr:produits-aoc": "protected-designation label, not a category (see en:aoc-cheeses).",
    "en:labeled-cheeses": (
        "quality-label attribute, not a category — 'this cheese carries some "
        "label' is not a place in a product hierarchy."
    ),
    "fr:produits-labellises": "quality-label attribute, not a category (see en:labeled-cheeses).",
    "en:proposed-for-deletion": "upstream taxonomy maintenance marker, not a product category.",
    "en:empty": "placeholder for a missing value.",
}


@dataclass(frozen=True)
class CategoryVocabulary:
    """The category ids a run is allowed to name, and the ids that merely exist.

    ``eligible`` is the set a catalog may put a product *under*: exactly
    ``taxonomy.eligible_nodes`` for this run's languages, which is the same set
    ``category_chain`` picks its leaf from, so the flat ``categories`` field and
    the tip of ``category_path`` cannot draw on different vocabularies.

    It is deliberately **not** the set a path may walk *through*: the parent map
    is language-blind, so a chain may pass through a foreign ancestor whose id
    would be refused as a flat value. That is the point rather than a leak — the
    path segment is the taxonomy's structure, which has no language, while a flat
    value is a searchable claim about the product, which does.

    ``known`` is every id in the loaded taxonomy, and exists only to tell an
    out-of-language node apart from a tag that is not in the taxonomy at all.
    """

    eligible: Set[str]
    known: Set[str]

    @classmethod
    def for_catalog(
        cls,
        taxonomy: Dict[str, Any],
        keep_prefixes: Optional[Set[str]],
        exclude: Optional[Set[str]] = None,
    ) -> "CategoryVocabulary":
        """Build from the taxonomy directly, via the one eligibility rule."""
        return cls(
            eligible=eligible_nodes(taxonomy, keep_prefixes=keep_prefixes, exclude=exclude),
            known=set(taxonomy),
        )


@dataclass
class CuratedTags:
    """One product's tags after aliasing, with every refusal and its reason."""

    accepted: List[str] = field(default_factory=list)
    rejected: List[Tuple[str, str]] = field(default_factory=list)
    aliased: int = 0

    @property
    def instances(self) -> int:
        return len(self.accepted) + len(self.rejected)


def curate_category_tags(
    tags: Sequence[str],
    vocabulary: Optional[CategoryVocabulary],
    exclude: Set[str],
) -> CuratedTags:
    """Alias, de-duplicate and validate one product's ``categories_tags``.

    Order is preserved — the first accepted tag is the product's primary
    category — and de-duplication happens *after* aliasing, so two ids that
    canonicalise to the same node (``en:melted-cheese`` and
    ``en:processed-cheese``) collapse to one.

    ``vocabulary`` of ``None`` means no taxonomy was loaded (``--no-taxonomy``).
    There is then no vocabulary to validate against, so tags are aliased and
    excluded but never refused: refusing everything would empty the flat
    ``categories`` field for the whole run, which is a worse outcome than the
    unvalidated field this module exists to replace. The curated drops still
    apply — they do not need a taxonomy to be wrong.
    """
    out = CuratedTags()
    seen: Set[str] = set()
    for raw in tags:
        if raw in exclude:
            continue
        tag = TAG_ALIASES.get(raw, raw)
        if tag != raw:
            out.aliased += 1
        if tag in seen or tag in exclude:
            continue
        seen.add(tag)
        if tag in TAG_DROPS:
            out.rejected.append((tag, REASON_CURATED_DROP))
        elif vocabulary is None or tag in vocabulary.eligible:
            out.accepted.append(tag)
        elif tag in vocabulary.known:
            out.rejected.append((tag, REASON_OUT_OF_LANGUAGE))
        else:
            out.rejected.append((tag, REASON_NOT_IN_TAXONOMY))
    return out


class TagCurationAudit:
    """Per-run totals for refused category tags, for the extraction report.

    The rate is the point. 4.26% of tag instances had no taxonomy node when this
    was measured by hand; without a number in the report, the next person to
    wonder has to reverse-map the built index against the taxonomy to find out,
    which is how this defect was found in the first place.
    """

    def __init__(self, top_n: int = 20) -> None:
        self.top_n = top_n
        self.products_with_tags = 0
        self.products_with_rejected_tags = 0
        self.products_with_no_accepted_tag = 0
        self.tag_instances = 0
        self.accepted_instances = 0
        self.aliased_instances = 0
        self.by_reason: Counter = Counter()
        self.rejected_tags: Dict[str, Counter] = {
            REASON_CURATED_DROP: Counter(),
            REASON_NOT_IN_TAXONOMY: Counter(),
            REASON_OUT_OF_LANGUAGE: Counter(),
        }

    def record(self, curated: CuratedTags) -> None:
        if not curated.instances:
            return
        self.products_with_tags += 1
        self.tag_instances += curated.instances
        self.accepted_instances += len(curated.accepted)
        self.aliased_instances += curated.aliased
        if curated.rejected:
            self.products_with_rejected_tags += 1
        if not curated.accepted:
            self.products_with_no_accepted_tag += 1
        for tag, reason in curated.rejected:
            self.by_reason[reason] += 1
            self.rejected_tags[reason][tag] += 1

    @property
    def rejected_instances(self) -> int:
        return sum(self.by_reason.values())

    @property
    def unknown_instances(self) -> int:
        return self.by_reason[REASON_NOT_IN_TAXONOMY]

    @property
    def distinct_unknown_tags(self) -> int:
        return len(self.rejected_tags[REASON_NOT_IN_TAXONOMY])

    def _top(self, reason: str) -> List[Dict[str, Any]]:
        return [
            {"tag": tag, "instances": n}
            for tag, n in self.rejected_tags[reason].most_common(self.top_n)
        ]

    def summary(self) -> Dict[str, Any]:
        rate = self.unknown_instances / self.tag_instances if self.tag_instances else 0.0
        rejected_rate = self.rejected_instances / self.tag_instances if self.tag_instances else 0.0
        return {
            "products_with_tags": self.products_with_tags,
            "products_with_rejected_tags": self.products_with_rejected_tags,
            "products_with_no_accepted_tag": self.products_with_no_accepted_tag,
            "tag_instances": self.tag_instances,
            "accepted_instances": self.accepted_instances,
            "aliased_instances": self.aliased_instances,
            "rejected_instances": self.rejected_instances,
            "rejected_rate": round(rejected_rate, 6),
            "unknown_tag_instances": self.unknown_instances,
            "unknown_tag_rate": round(rate, 6),
            "distinct_unknown_tags": self.distinct_unknown_tags,
            "rejected_by_reason": dict(sorted(self.by_reason.items())),
            "top_unknown_tags": self._top(REASON_NOT_IN_TAXONOMY),
            "top_out_of_language_tags": self._top(REASON_OUT_OF_LANGUAGE),
            "curated_drops": [
                {"tag": tag, "instances": n, "reason": TAG_DROPS.get(tag, "")}
                for tag, n in self.rejected_tags[REASON_CURATED_DROP].most_common()
            ],
        }

    def log_lines(self) -> List[str]:
        """One-screen summary for stderr at the end of a run."""
        s = self.summary()
        if not s["tag_instances"]:
            return ["Category tags: no product carried a category tag."]
        lines = [
            "Category tags: "
            f"{s['accepted_instances']:,}/{s['tag_instances']:,} accepted, "
            f"{s['rejected_instances']:,} refused ({s['rejected_rate']:.2%}), "
            f"{s['aliased_instances']:,} aliased to a renamed node.",
            f"  no taxonomy node: {s['unknown_tag_instances']:,} instances "
            f"({s['unknown_tag_rate']:.2%}), {s['distinct_unknown_tags']:,} distinct.",
            f"  products left with no usable category: "
            f"{s['products_with_no_accepted_tag']:,}/{s['products_with_tags']:,}.",
        ]
        top = ", ".join(f"{t['tag']} x{t['instances']:,}" for t in s["top_unknown_tags"][:5])
        if top:
            lines.append(f"  top unresolved: {top}")
        return lines
