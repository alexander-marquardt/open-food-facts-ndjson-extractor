"""Unit tests for the curated alias/drop tables and the tag curation function.

The tables in ``off_demo_extract.category_tags`` are hand-authored data, and
hand-authored data is where a silent contradiction hides: an alias pointing at a
tag that is itself aliased, an alias pointing at a tag the drop list refuses, a
drop with no recorded reason. None of those would fail an extraction run — they
would just quietly produce a category vocabulary nobody intended. These pin the
invariants the tables have to satisfy, and the order in which the two are
applied.

Run with ``pytest tests/`` or directly: ``python tests/test_category_tag_curation.py``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from off_demo_extract.category_tags import (  # noqa: E402
    REASON_CURATED_DROP,
    REASON_NOT_IN_TAXONOMY,
    REASON_OUT_OF_LANGUAGE,
    TAG_ALIASES,
    TAG_DROPS,
    CategoryVocabulary,
    TagCurationAudit,
    curate_category_tags,
)

EXCLUDE = {"en:null", "en:unknown", "en:undefined"}

# A miniature vocabulary standing in for a loaded taxonomy: ``eligible`` is what
# a path may walk, ``known`` adds a node in a language this catalog does not use.
VOCAB = CategoryVocabulary(
    eligible={
        "en:snacks",
        "en:salty-snacks",
        "en:cheeses",
        "en:processed-cheeses",
        "en:food-decorations",
    },
    known={
        "en:snacks",
        "en:salty-snacks",
        "en:cheeses",
        "en:processed-cheeses",
        "en:food-decorations",
        "fr:charcuteries-cuites",
    },
)

TAG_ID = re.compile(r"^[a-z]{2,3}:[a-z0-9][a-z0-9-]*$")


def test_alias_targets_are_never_themselves_aliased() -> None:
    """No alias chains: one hop resolves a tag, or the map is wrong.

    ``curate_category_tags`` deliberately applies exactly one hop — a chain
    would make the result depend on iteration order, which is not something a
    data table should decide.
    """
    chained = {src: dst for src, dst in TAG_ALIASES.items() if dst in TAG_ALIASES}
    assert not chained, f"alias targets that are themselves aliased: {chained}"


def test_alias_targets_are_not_on_the_drop_list() -> None:
    """Aliasing a tag onto a tag we refuse would rename it into the bin."""
    contradictions = {src: dst for src, dst in TAG_ALIASES.items() if dst in TAG_DROPS}
    assert not contradictions, f"aliases pointing at dropped tags: {contradictions}"


def test_a_tag_is_never_both_aliased_and_dropped() -> None:
    both = sorted(set(TAG_ALIASES) & set(TAG_DROPS))
    assert not both, f"tags in both tables: {both}"


def test_every_drop_records_a_reason() -> None:
    """'Dropped explicitly, with the reason recorded' is the whole point."""
    missing = [tag for tag, reason in TAG_DROPS.items() if not reason.strip()]
    assert not missing, f"curated drops with no reason: {missing}"
    assert "en:groceries" in TAG_DROPS, "the largest single offender must be explicit"


def test_table_keys_are_well_formed_tag_ids() -> None:
    """A typo'd id is inert — it would never match, and nothing would say so."""
    bad = [t for t in [*TAG_ALIASES, *TAG_ALIASES.values(), *TAG_DROPS] if not TAG_ID.match(t)]
    assert not bad, f"malformed tag ids: {bad}"


def test_alias_is_applied_before_the_unknown_check() -> None:
    """The renamed id must survive; checking membership first would refuse it."""
    curated = curate_category_tags(["en:salted-snacks"], VOCAB, EXCLUDE)
    assert curated.accepted == ["en:salty-snacks"], curated
    assert curated.rejected == []
    assert curated.aliased == 1


def test_unknown_tag_is_refused_but_its_record_keeps_the_rest() -> None:
    """Drop the value, never the record — the 87% case."""
    curated = curate_category_tags(
        ["en:cheeses", "en:not-a-taxonomy-node"], VOCAB, EXCLUDE
    )
    assert curated.accepted == ["en:cheeses"]
    assert curated.rejected == [("en:not-a-taxonomy-node", REASON_NOT_IN_TAXONOMY)]


def test_the_three_refusal_reasons_are_told_apart() -> None:
    """Each refusal carries why, so the report can separate policy from tail."""
    curated = curate_category_tags(
        [
            "en:cheeses",
            "en:groceries",
            "fr:charcuteries-cuites",
            "en:invented-by-a-contributor",
        ],
        VOCAB,
        EXCLUDE,
    )
    assert curated.accepted == ["en:cheeses"]
    assert dict(curated.rejected) == {
        "en:groceries": REASON_CURATED_DROP,
        "fr:charcuteries-cuites": REASON_OUT_OF_LANGUAGE,
        "en:invented-by-a-contributor": REASON_NOT_IN_TAXONOMY,
    }


def test_aliasing_can_collapse_two_tags_into_one() -> None:
    """De-duplication happens after aliasing, or the same node lands twice."""
    curated = curate_category_tags(
        ["en:melted-cheese", "en:processed-cheese"], VOCAB, EXCLUDE
    )
    assert curated.accepted == ["en:processed-cheeses"], curated
    assert curated.aliased == 2


def test_excluded_sentinels_are_neither_accepted_nor_counted() -> None:
    """``en:undefined`` is already handled by --category-exclude; it is not a defect."""
    curated = curate_category_tags(["en:undefined", "en:cheeses"], VOCAB, EXCLUDE)
    assert curated.accepted == ["en:cheeses"]
    assert curated.rejected == []
    assert curated.instances == 1


def test_without_a_taxonomy_tags_are_kept_but_curated_drops_still_apply() -> None:
    """--no-taxonomy has nothing to validate against, so it must not refuse everything.

    Refusing every tag with no vocabulary loaded would empty the flat
    ``categories`` field for the entire run — worse than the unvalidated field
    this module replaces. The curated drops do not need a taxonomy to be wrong,
    so they still fire.
    """
    curated = curate_category_tags(
        ["en:cheeses", "en:groceries", "en:whatever-this-is"], None, EXCLUDE
    )
    assert curated.accepted == ["en:cheeses", "en:whatever-this-is"]
    assert curated.rejected == [("en:groceries", REASON_CURATED_DROP)]


def test_audit_totals_and_top_n() -> None:
    """The audit is what makes the rate visible per run rather than by hand."""
    audit = TagCurationAudit(top_n=2)
    for tags in (
        ["en:cheeses", "en:groceries", "en:tail-a"],
        ["en:cheeses", "en:tail-a"],
        ["en:tail-b"],
    ):
        audit.record(curate_category_tags(tags, VOCAB, EXCLUDE))

    summary = audit.summary()
    assert summary["tag_instances"] == 6
    assert summary["accepted_instances"] == 2
    assert summary["rejected_instances"] == 4
    assert summary["unknown_tag_instances"] == 3
    assert summary["distinct_unknown_tags"] == 2
    assert summary["products_with_tags"] == 3
    assert summary["products_with_rejected_tags"] == 3
    assert summary["products_with_no_accepted_tag"] == 1
    assert summary["rejected_by_reason"] == {REASON_CURATED_DROP: 1, REASON_NOT_IN_TAXONOMY: 3}
    assert summary["top_unknown_tags"] == [
        {"tag": "en:tail-a", "instances": 2},
        {"tag": "en:tail-b", "instances": 1},
    ]
    assert summary["curated_drops"] == [
        {"tag": "en:groceries", "instances": 1, "reason": TAG_DROPS["en:groceries"]}
    ]
    assert abs(summary["unknown_tag_rate"] - 0.5) < 1e-9


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
