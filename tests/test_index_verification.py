"""Tests for ``scripts/verify_index.py`` — the index-side half of build verification.

Two things make these tests worth trusting rather than merely green.

**The response envelopes are live captures.** Every Elasticsearch response these
tests replay comes from ``tests/fixtures/index_verification_envelopes.json``,
which was recorded against a real ``catalog_es_v14`` index with the exact requests
``verify_index.py`` issues — a ``_mapping`` read, one ``size: 0`` search carrying
the coverage filter and both terms aggregations, and both shapes a ``composite``
aggregation can return (a page with an ``after_key``, and the terminal page with
neither buckets nor an ``after_key``). Only the *bucket contents* are varied per
scenario; the envelope — key names, nesting, ``hits.total.value``,
``sum_other_doc_count``, ``doc_count_error_upper_bound``, ``after_key`` — is the
cluster's own. A test built on an invented envelope proves the verifier agrees
with the test author, not with Elasticsearch. ``test_captured_envelopes_carry_the
_keys_the_verifier_reads`` pins that down so a later hand-edit of the fixture
cannot quietly drift away from the real shape.

**The scenarios are the ones that actually happened.** The numbers in the
count and run-profile tests are the measured ones: ``catalog_es_v13`` reconciles
exactly, ``catalog_en_v13`` is 928 documents short spread over 700 short runs,
and ``catalog_fr_v13`` is 27,746 short concentrated in 48 runs of which 13 hold
99.9%. Those two shortfalls have different causes, and a check that only reports
"short by N" cannot tell them apart — so the run profile is tested, not just the
total.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_index import (  # noqa: E402
    ReadOnlyClient,
    VerificationError,
    check_document_identity,
    contiguous_runs,
    infer_lang,
    terms_truncation,
    verify,
)

ENVELOPES = json.loads(
    (Path(__file__).parent / "fixtures" / "index_verification_envelopes.json").read_text(
        encoding="utf-8"
    )
)


# --------------------------------------------------------------------------- #
# a miniature world: three-node taxonomy, one manifest, one replayed cluster
# --------------------------------------------------------------------------- #


TAXONOMY = {
    "en:beverages": {"name": {"en": "Beverages"}, "parents": []},
    "en:hot-beverages": {"name": {"en": "Hot beverages"}, "parents": ["en:beverages"]},
    "en:teas": {"name": {"en": "Teas"}, "parents": ["en:hot-beverages"]},
}

DUMP_SHA = "f06f34f7ecd19405bf3e91a31d638d96ba91cd364bee69f9530a6c6380dd2f5f"
COMMIT = "9e8a4c6f38e7f03d7f42dc4fbc97601210285d83"


def write_taxonomy(tmp_path: Path, nodes: Optional[Dict[str, Any]] = None) -> Path:
    path = tmp_path / "categories.json"
    path.write_text(json.dumps(nodes if nodes is not None else TAXONOMY), encoding="utf-8")
    return path


def write_manifest(
    tmp_path: Path,
    *,
    lang: str = "en",
    distinct_ids: int = 3,
    written: Optional[int] = None,
    taxonomy_sha: str = "",
    name: str = "build_manifest.json",
    artifact_verification: Any = "default",
) -> Path:
    entry: Dict[str, Any] = {
        "lang": lang,
        "counters": {"written": distinct_ids if written is None else written},
    }
    if artifact_verification == "default":
        entry["artifact_verification"] = {"distinct_ids": distinct_ids}
    elif artifact_verification is not None:
        entry["artifact_verification"] = artifact_verification
    manifest = {
        "schema": "off-catalog-build-manifest/1",
        "generated_utc": "2026-08-03T10:34:52Z",
        "extractor": {"commit": COMMIT},
        "dump": {"sha256": DUMP_SHA},
        "taxonomy": {"sha256": taxonomy_sha},
        "locales": [entry],
    }
    path = tmp_path / name
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _buckets(counts: Dict[str, int]) -> List[Dict[str, Any]]:
    return [{"key": key, "doc_count": count} for key, count in counts.items()]


def search_response(
    *,
    total: int,
    with_path: int,
    taxonomy_tags: Dict[str, int],
    category_path: Dict[str, int],
    sum_other_doc_count: int = 0,
    doc_count_error_upper_bound: int = 0,
) -> Dict[str, Any]:
    """A live-captured search envelope with this scenario's buckets dropped in."""
    response = copy.deepcopy(ENVELOPES["search_response"])
    response["hits"]["total"]["value"] = total
    response["aggregations"]["with_category_path"]["doc_count"] = with_path
    for name, counts in (("taxonomy_tags", taxonomy_tags), ("category_path", category_path)):
        agg = response["aggregations"][name]
        agg["buckets"] = _buckets(counts)
        agg["sum_other_doc_count"] = sum_other_doc_count
        agg["doc_count_error_upper_bound"] = doc_count_error_upper_bound
    return response


def composite_pages(values: Dict[str, int]) -> List[Dict[str, Any]]:
    """One full page plus the terminal page, both on captured envelopes."""
    page = copy.deepcopy(ENVELOPES["composite_page_response"])
    agg = page["aggregations"]["values"]
    agg["buckets"] = [{"key": {"v": key}, "doc_count": count} for key, count in values.items()]
    if values:
        agg["after_key"] = {"v": list(values)[-1]}
    else:
        agg.pop("after_key", None)
    return [page, copy.deepcopy(ENVELOPES["composite_terminal_response"])]


class ReplayClient:
    """Serves captured envelopes, and records every request the verifier makes."""

    def __init__(
        self,
        search: Dict[str, Any],
        *,
        meta: Optional[Dict[str, Any]] = None,
        composite: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> None:
        self.mapping = copy.deepcopy(ENVELOPES["mapping_response"])
        if meta is not None:
            next(iter(self.mapping.values()))["mappings"]["_meta"] = meta
        self.search = search
        self.composite = {field: list(pages) for field, pages in (composite or {}).items()}
        self.requests: List[str] = []
        self.bodies: List[Optional[Dict[str, Any]]] = []

    def request(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.requests.append(path)
        self.bodies.append(body)
        if path.endswith("/_mapping"):
            index = path.strip("/").split("/")[0]
            return {index: next(iter(self.mapping.values()))}
        source = ((body or {}).get("aggs") or {}).get("values") or {}
        if "composite" in source:
            field = source["composite"]["sources"][0]["v"]["terms"]["field"]
            pages = self.composite.get(field)
            if not pages:
                raise AssertionError(f"unscripted composite enumeration of {field!r}")
            return pages.pop(0)
        return self.search

    @property
    def composite_fields(self) -> List[str]:
        fields = []
        for body in self.bodies:
            source = ((body or {}).get("aggs") or {}).get("values") or {}
            if "composite" in source:
                fields.append(source["composite"]["sources"][0]["v"]["terms"]["field"])
        return fields


def run(client: ReplayClient, manifest: Path, **kwargs: Any) -> Dict[str, Any]:
    return verify(client, "catalog_en_v13", manifest, **kwargs)


def check(result: Dict[str, Any], name: str) -> Dict[str, Any]:
    return next(c for c in result["checks"] if c["check"] == name)


# --------------------------------------------------------------------------- #
# the fixture is a live capture, and has to stay one
# --------------------------------------------------------------------------- #


def test_captured_envelopes_carry_the_keys_the_verifier_reads() -> None:
    search = ENVELOPES["search_response"]
    assert search["hits"]["total"]["relation"] == "eq", "track_total_hits must give an exact total"
    assert isinstance(search["hits"]["total"]["value"], int)
    assert "doc_count" in search["aggregations"]["with_category_path"]
    for name in ("taxonomy_tags", "category_path"):
        agg = search["aggregations"][name]
        assert {"buckets", "sum_other_doc_count", "doc_count_error_upper_bound"} <= set(agg)
        assert {"key", "doc_count"} <= set(agg["buckets"][0])

    page = ENVELOPES["composite_page_response"]["aggregations"]["values"]
    assert "after_key" in page and page["buckets"][0]["key"].keys() == {"v"}
    terminal = ENVELOPES["composite_terminal_response"]["aggregations"]["values"]
    assert terminal["buckets"] == [] and "after_key" not in terminal


def test_the_captured_mapping_declares_the_field_the_verifier_aggregates_on() -> None:
    """The two halves of the rename, tied together inside one capture.

    A ``terms`` aggregation on a field the mapping does not have returns zero
    buckets rather than an error, so a verifier pointed at the wrong name reports
    "no values outside the snapshot" for the one reason that proves nothing. On
    ``catalog_en_v14`` the old name returns 0 buckets against 108,379 documents
    and ``taxonomy_tags`` returns real ones; nothing in the response says which
    of the two happened.

    The mapping and the aggregation come from the same captured index, so this
    is the assertion a half-done rename fails: moving the aggregation key without
    the field it names, or the reverse, no longer agrees with the capture.
    """
    properties = next(iter(ENVELOPES["mapping_response"].values()))["mappings"]["properties"]
    assert "taxonomy_tags" in properties
    assert "categories" not in properties
    assert set(ENVELOPES["search_response"]["aggregations"]) - {"with_category_path"} <= set(
        properties
    )


def test_the_live_index_records_no_build_identity() -> None:
    """The premise of the ``manifest_identity`` check, pinned against reality.

    If a loader ever starts writing ``_meta``, this test fails and the capture —
    and the check's ``unverifiable`` verdict — need revisiting. That is the right
    outcome: the verdict is a statement about the world, not a constant.
    """
    mappings = next(iter(ENVELOPES["mapping_response"].values()))["mappings"]
    assert "_meta" not in mappings


# --------------------------------------------------------------------------- #
# document count
# --------------------------------------------------------------------------- #


def test_count_reconciles_when_the_index_matches_distinct_ids(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, distinct_ids=31910)
    client = ReplayClient(
        search_response(total=31910, with_path=31910, taxonomy_tags={}, category_path={})
    )
    result = run(client, manifest)
    assert check(result, "document_count")["status"] == "pass"
    assert result["failed"] == []


def test_count_fails_on_the_measured_fr_shortfall(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, distinct_ids=222955, written=223036)
    client = ReplayClient(
        search_response(total=195209, with_path=195209, taxonomy_tags={}, category_path={})
    )
    result = run(client, manifest)
    count = check(result, "document_count")
    assert count["status"] == "fail"
    assert count["indexed_documents"] == 195209
    assert count["manifest_distinct_ids"] == 222955
    assert count["delta"] == -27746
    assert "document_count" in result["failed"]


def test_count_uses_distinct_ids_and_not_the_record_count(tmp_path: Path) -> None:
    """An index holding exactly ``distinct_ids`` documents is correct.

    The fr catalog holds 223,036 records for 222,955 distinct ids. Reconciling
    against records would call a correct index 81 documents short — which is the
    kind of small unexplained delta that gets rationalised away, and then the
    large one does too.
    """
    manifest = write_manifest(tmp_path, distinct_ids=222955, written=223036)
    client = ReplayClient(
        search_response(total=222955, with_path=222955, taxonomy_tags={}, category_path={})
    )
    assert check(run(client, manifest), "document_count")["status"] == "pass"

    against_records = ReplayClient(
        search_response(total=223036, with_path=223036, taxonomy_tags={}, category_path={})
    )
    assert check(run(against_records, manifest), "document_count")["status"] == "fail"


def test_manifest_without_distinct_ids_fails_loudly(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, artifact_verification=None)
    client = ReplayClient(search_response(total=3, with_path=3, taxonomy_tags={}, category_path={}))
    with pytest.raises(VerificationError, match="artifact_verification.distinct_ids"):
        run(client, manifest)


def test_unknown_locale_is_named(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, lang="es")
    client = ReplayClient(search_response(total=3, with_path=3, taxonomy_tags={}, category_path={}))
    with pytest.raises(VerificationError, match="no locale 'en'"):
        run(client, manifest)


# --------------------------------------------------------------------------- #
# category_path coverage — the other way a count can be short
# --------------------------------------------------------------------------- #


def test_documents_present_without_category_path_are_reported(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, distinct_ids=100)
    client = ReplayClient(
        search_response(total=100, with_path=72, taxonomy_tags={}, category_path={})
    )
    coverage = check(run(client, manifest), "category_path_coverage")
    assert coverage["status"] == "fail"
    assert coverage["without_category_path"] == 28


def test_full_coverage_separates_a_missing_load_from_a_missed_update(tmp_path: Path) -> None:
    """The measurement that resolved the fr question.

    Every one of the 195,209 fr documents carries ``category_path``, so no
    document was written and then skipped by the partial-update pass. Combined
    with the id-set check finding zero index ids absent from the extract, that
    leaves only one mechanism: the missing documents were never written.
    """
    manifest = write_manifest(tmp_path, distinct_ids=222955)
    client = ReplayClient(
        search_response(total=195209, with_path=195209, taxonomy_tags={}, category_path={})
    )
    result = run(client, manifest)
    assert check(result, "category_path_coverage")["status"] == "pass"
    assert check(result, "document_count")["status"] == "fail"


# --------------------------------------------------------------------------- #
# manifest identity
# --------------------------------------------------------------------------- #


def test_identity_is_unverifiable_when_the_index_says_nothing(tmp_path: Path) -> None:
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=0, taxonomy_sha=_sha(taxonomy))
    client = ReplayClient(
        search_response(total=0, with_path=0, taxonomy_tags={}, category_path={})
    )
    result = run(client, manifest, taxonomy_path=taxonomy)
    identity = check(result, "manifest_identity")
    assert identity["status"] == "unverifiable"
    assert "_meta" in identity["remedy"]
    assert identity["asserted"]["dump_sha256"] == DUMP_SHA
    # Unverifiable is not failure: the index may be perfectly correct, we simply
    # cannot ask it. Turning that into a red would train operators to ignore it.
    assert result["failed"] == []
    assert result["unverifiable"] == ["manifest_identity"]


def test_identity_fails_when_the_index_claims_a_different_build(tmp_path: Path) -> None:
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=0, taxonomy_sha=_sha(taxonomy))
    client = ReplayClient(
        search_response(total=0, with_path=0, taxonomy_tags={}, category_path={}),
        meta={
            "off_catalog_build": {
                "lang": "en",
                "extractor_commit": COMMIT,
                "dump_sha256": "0" * 64,
                "taxonomy_sha256": _sha(taxonomy),
            }
        },
    )
    result = run(client, manifest, taxonomy_path=taxonomy)
    identity = check(result, "manifest_identity")
    assert identity["status"] == "fail"
    assert identity["disagreements"]["dump_sha256"]["manifest_says"] == DUMP_SHA


def test_identity_accepts_an_index_that_agrees_with_the_manifest(tmp_path: Path) -> None:
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=0, taxonomy_sha=_sha(taxonomy))
    client = ReplayClient(
        search_response(total=0, with_path=0, taxonomy_tags={}, category_path={}),
        meta={
            "off_catalog_build": {
                "lang": "en",
                "extractor_commit": COMMIT,
                "dump_sha256": DUMP_SHA,
                "taxonomy_sha256": _sha(taxonomy),
            }
        },
    )
    assert check(run(client, manifest, taxonomy_path=taxonomy), "manifest_identity")["status"] == (
        "pass"
    )


def test_a_taxonomy_file_that_is_not_the_pinned_snapshot_is_refused(tmp_path: Path) -> None:
    """Otherwise the vocabulary checks silently judge the index against the wrong snapshot."""
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=0, taxonomy_sha="deadbeef" * 8)
    client = ReplayClient(
        search_response(total=0, with_path=0, taxonomy_tags={}, category_path={})
    )
    result = run(client, manifest, taxonomy_path=taxonomy)
    identity = check(result, "manifest_identity")
    assert identity["status"] == "fail"
    assert identity["taxonomy_file_sha256"] == _sha(taxonomy)


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# truncation
# --------------------------------------------------------------------------- #


def test_saturated_terms_aggregation_escalates_to_composite(tmp_path: Path) -> None:
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=2, taxonomy_sha=_sha(taxonomy))
    everything = {"Beverages": 2, "Hot beverages": 1, "Teas": 1}
    client = ReplayClient(
        search_response(
            total=2,
            with_path=2,
            taxonomy_tags={"Beverages": 2},
            category_path={"Beverages": 2},
            sum_other_doc_count=1,
        ),
        composite={
            "taxonomy_tags": composite_pages(everything),
            "category_path": composite_pages(
                {"Beverages": 2, "Beverages/Hot beverages": 1, "Beverages/Hot beverages/Teas": 1}
            ),
        },
    )
    result = run(client, manifest, taxonomy_path=taxonomy, terms_size=1)
    truncation = result["aggregation_truncation"]["taxonomy_tags"]
    assert truncation["size_saturated"] is True
    assert truncation["escalated_to_composite"] is True
    assert truncation["exhaustive_distinct_values"] == 3
    vocabulary = check(result, "category_vocabulary")
    assert vocabulary["distinct_categories_in_index"] == 3
    assert vocabulary["status"] == "pass"


def test_saturation_escalates_even_when_sum_other_doc_count_is_zero(tmp_path: Path) -> None:
    """The multi-valued-field hazard, made explicit.

    ``taxonomy_tags`` and ``category_path`` are both multi-valued, so a document is
    counted in every bucket it lands in and the returned bucket doc counts can
    already account for the whole index. ``sum_other_doc_count`` then reads zero
    while terms are still missing. A verifier that trusted it would report "no
    values outside the snapshot" precisely because it had not seen them — the
    same silent under-report this whole script exists to prevent.
    """
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=2, taxonomy_sha=_sha(taxonomy))
    client = ReplayClient(
        search_response(
            total=2,
            with_path=2,
            taxonomy_tags={"Beverages": 2},
            category_path={"Beverages": 2},
            sum_other_doc_count=0,
            doc_count_error_upper_bound=0,
        ),
        composite={
            "taxonomy_tags": composite_pages({"Beverages": 2, "Not in the snapshot": 1}),
            "category_path": composite_pages({"Beverages": 2}),
        },
    )
    result = run(client, manifest, taxonomy_path=taxonomy, terms_size=1)
    assert result["aggregation_truncation"]["taxonomy_tags"]["escalated_to_composite"] is True
    vocabulary = check(result, "category_vocabulary")
    assert vocabulary["values_outside_snapshot"] == 1
    assert vocabulary["status"] == "fail"


def test_unsaturated_terms_aggregation_does_not_pay_for_composite(tmp_path: Path) -> None:
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=2, taxonomy_sha=_sha(taxonomy))
    client = ReplayClient(
        search_response(
            total=2, with_path=2, taxonomy_tags={"Beverages": 2}, category_path={"Beverages": 2}
        )
    )
    result = run(client, manifest, taxonomy_path=taxonomy, terms_size=100)
    assert client.composite_fields == []
    # One mapping read and one search: the whole default verification.
    assert client.requests == ["/catalog_en_v13/_mapping", "/catalog_en_v13/_search"]
    assert result["aggregation_truncation"]["taxonomy_tags"]["escalated_to_composite"] is False


def test_terms_truncation_reads_all_three_signals() -> None:
    assert terms_truncation({"buckets": [{"key": "a"}], "sum_other_doc_count": 0}, 10)["complete"]
    assert not terms_truncation({"buckets": [{"key": "a"}], "sum_other_doc_count": 4}, 10)[
        "complete"
    ]
    assert not terms_truncation(
        {"buckets": [{"key": "a"}], "sum_other_doc_count": 0, "doc_count_error_upper_bound": 2}, 10
    )["complete"]
    assert not terms_truncation({"buckets": [{"key": "a"}], "sum_other_doc_count": 0}, 1)[
        "complete"
    ]


# --------------------------------------------------------------------------- #
# vocabulary, in both directions
# --------------------------------------------------------------------------- #


def test_vocabulary_reports_values_the_snapshot_does_not_explain(tmp_path: Path) -> None:
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=2, taxonomy_sha=_sha(taxonomy))
    client = ReplayClient(
        search_response(
            total=2,
            with_path=2,
            # "Plant based foods" is the real shape of the live drift: the
            # snapshot renders the label with a hyphen, the index has an older
            # rendering, and a set difference is the only thing that notices.
            taxonomy_tags={"Beverages": 2, "Plant based foods": 5},
            category_path={"Beverages": 2},
        ),
    )
    vocabulary = check(run(client, manifest, taxonomy_path=taxonomy), "category_vocabulary")
    assert vocabulary["status"] == "fail"
    assert vocabulary["values_outside_snapshot"] == 1
    assert vocabulary["value_instances_outside_snapshot"] == 5
    assert vocabulary["top_values_outside_snapshot"][0]["value"] == "Plant based foods"


def test_vocabulary_reports_snapshot_labels_the_index_never_uses(tmp_path: Path) -> None:
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=2, taxonomy_sha=_sha(taxonomy))
    client = ReplayClient(
        search_response(
            total=2, with_path=2, taxonomy_tags={"Beverages": 2}, category_path={"Beverages": 2}
        ),
    )
    vocabulary = check(run(client, manifest, taxonomy_path=taxonomy), "category_vocabulary")
    assert vocabulary["snapshot_labels_unused_by_index"] == 2
    assert set(vocabulary["sample_snapshot_labels_unused_by_index"]) == {"Hot beverages", "Teas"}
    # Unused labels are normal: a catalog need not exercise the whole taxonomy.
    assert vocabulary["status"] == "pass"


def test_vocabulary_flags_a_path_head_that_is_not_a_taxonomy_root(tmp_path: Path) -> None:
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=1, taxonomy_sha=_sha(taxonomy))
    client = ReplayClient(
        search_response(
            total=1,
            with_path=1,
            taxonomy_tags={"Teas": 1},
            category_path={"Hot beverages": 1, "Hot beverages/Teas": 1},
        ),
    )
    vocabulary = check(run(client, manifest, taxonomy_path=taxonomy), "category_vocabulary")
    assert vocabulary["status"] == "fail"
    assert vocabulary["path_heads_outside_taxonomy_roots"] == 1
    assert vocabulary["top_path_heads_outside_taxonomy_roots"] == ["Hot beverages"]


def test_vocabulary_flags_an_address_whose_parent_never_reached_the_index(
    tmp_path: Path,
) -> None:
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=1, taxonomy_sha=_sha(taxonomy))
    client = ReplayClient(
        search_response(
            total=1,
            with_path=1,
            taxonomy_tags={"Teas": 1},
            category_path={"Beverages": 1, "Beverages/Hot beverages/Teas": 1},
        ),
    )
    vocabulary = check(run(client, manifest, taxonomy_path=taxonomy), "category_vocabulary")
    assert vocabulary["status"] == "fail"
    assert vocabulary["orphan_addresses"] == 1
    assert vocabulary["top_orphan_addresses"] == ["Beverages/Hot beverages/Teas"]


def test_vocabulary_flags_a_path_segment_absent_from_the_snapshot(tmp_path: Path) -> None:
    taxonomy = write_taxonomy(tmp_path)
    manifest = write_manifest(tmp_path, distinct_ids=1, taxonomy_sha=_sha(taxonomy))
    client = ReplayClient(
        search_response(
            total=1,
            with_path=1,
            taxonomy_tags={"Beverages": 1},
            category_path={"Beverages": 1, "Beverages/Invented": 1},
        ),
    )
    vocabulary = check(run(client, manifest, taxonomy_path=taxonomy), "category_vocabulary")
    assert vocabulary["status"] == "fail"
    assert vocabulary["top_path_segments_outside_snapshot"] == ["Invented"]


def test_vocabulary_is_skipped_rather_than_faked_without_a_taxonomy(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, distinct_ids=1)
    client = ReplayClient(
        search_response(total=1, with_path=1, taxonomy_tags={"anything": 1}, category_path={})
    )
    result = run(client, manifest)
    assert check(result, "category_vocabulary")["status"] == "skipped"
    assert result["failed"] == []


# --------------------------------------------------------------------------- #
# id sets and run profiles
# --------------------------------------------------------------------------- #


def write_catalog(tmp_path: Path, ids: List[str], name: str = "catalog.ndjson") -> Path:
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps({"id": identifier}) + "\n" for identifier in ids), encoding="utf-8"
    )
    return path


def test_run_profile_separates_scattered_drift_from_dropped_batches(tmp_path: Path) -> None:
    """The discriminator, on the two real shapes.

    en is short by 928 ids scattered over 700 runs — the signature of two
    extracts that disagree record by record. fr is short by ids that arrive in a
    few long contiguous stretches — the signature of a load whose batches went
    missing. The totals alone say only "short".
    """
    catalog_ids = [f"{n:07d}" for n in range(1000)]
    catalog = write_catalog(tmp_path, catalog_ids)

    scattered = [i for n, i in enumerate(catalog_ids) if n % 10 != 3]
    client = ReplayClient(search_response(total=0, with_path=0, taxonomy_tags={}, category_path={}))
    client.composite = {"id": composite_pages({i: 1 for i in scattered})}
    identity = check_document_identity(client.request, "catalog_en_v13", catalog)
    assert identity["status"] == "fail"
    assert identity["catalog_ids_absent_from_index"] == 100
    assert identity["missing_contiguous_runs"] == 100
    assert identity["missing_runs_of_100_or_more"] == 0

    blocked = catalog_ids[:400] + catalog_ids[600:]
    client.composite = {"id": composite_pages({i: 1 for i in blocked})}
    identity = check_document_identity(client.request, "catalog_en_v13", catalog)
    assert identity["catalog_ids_absent_from_index"] == 200
    assert identity["missing_contiguous_runs"] == 1
    assert identity["missing_runs_of_100_or_more"] == 1
    assert identity["ids_in_runs_of_100_or_more"] == 200
    assert identity["largest_missing_runs"][0] == {"catalog_position": 400, "length": 200}


def test_identity_reports_index_documents_the_catalog_does_not_have(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path, ["a", "b"])
    client = ReplayClient(search_response(total=0, with_path=0, taxonomy_tags={}, category_path={}))
    client.composite = {"id": composite_pages({"a": 1, "b": 1, "stranger": 1})}
    identity = check_document_identity(client.request, "catalog_en_v13", catalog)
    assert identity["status"] == "fail"
    assert identity["index_ids_absent_from_catalog"] == 1
    assert identity["sample_index_ids_absent_from_catalog"] == ["stranger"]


def test_identity_passes_when_both_sides_hold_the_same_ids(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path, ["a", "b", "c"])
    client = ReplayClient(search_response(total=0, with_path=0, taxonomy_tags={}, category_path={}))
    client.composite = {"id": composite_pages({"c": 1, "a": 1, "b": 1})}
    identity = check_document_identity(client.request, "catalog_en_v13", catalog)
    assert identity["status"] == "pass"
    assert identity["catalog_ids_absent_from_index"] == 0


def test_catalog_duplicate_ids_are_counted_once(tmp_path: Path) -> None:
    """Same rule as the count check: one document per distinct id, not per record."""
    catalog = write_catalog(tmp_path, ["a", "b", "a"])
    client = ReplayClient(search_response(total=0, with_path=0, taxonomy_tags={}, category_path={}))
    client.composite = {"id": composite_pages({"a": 1, "b": 1})}
    identity = check_document_identity(client.request, "catalog_en_v13", catalog)
    assert identity["catalog_distinct_ids"] == 2
    assert identity["status"] == "pass"


def test_contiguous_runs_collapses_adjacent_positions() -> None:
    assert contiguous_runs([]) == []
    assert contiguous_runs([4]) == [(4, 4)]
    assert contiguous_runs([0, 1, 2, 7, 9, 10]) == [(0, 2), (7, 7), (9, 10)]


# --------------------------------------------------------------------------- #
# read-only-ness is a property of the code, not of the author's intentions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "path",
    [
        "/catalog_en_v13/_bulk",
        "/catalog_en_v13/_update/123",
        "/catalog_en_v13/_delete_by_query",
        "/catalog_en_v13/_doc/123",
        "/catalog_en_v13",
        "/_reindex",
        "/catalog_en_v13/_search/../_bulk",
    ],
)
def test_the_client_refuses_every_endpoint_that_could_write(path: str) -> None:
    client = ReadOnlyClient("https://example.invalid", "unused")
    with pytest.raises(VerificationError, match="read-only"):
        client.request(path)
    assert client.requests == []


@pytest.mark.parametrize(
    "path",
    [
        "/catalog_en_v13/_search",
        "/catalog_en_v13/_count",
        "/catalog_en_v13/_mapping",
        "/catalog_en_v13/_settings",
    ],
)
def test_the_allowlist_admits_the_endpoints_the_verifier_needs(path: str) -> None:
    endpoint = path.rsplit("/", 1)[-1]
    from verify_index import READ_ONLY_ENDPOINTS

    assert endpoint in READ_ONLY_ENDPOINTS


def test_the_api_key_is_never_read_from_anywhere_but_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRISM_ELASTICSEARCH_API_KEY", raising=False)
    monkeypatch.setenv("PRISM_ELASTICSEARCH_URL", "https://example.invalid")
    with pytest.raises(VerificationError, match="must be set"):
        ReadOnlyClient.from_env()


def test_only_the_two_default_requests_are_issued(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, distinct_ids=1)
    client = ReplayClient(
        search_response(total=1, with_path=1, taxonomy_tags={}, category_path={})
    )
    result = run(client, manifest)
    assert result["requests"] == ["/catalog_en_v13/_mapping", "/catalog_en_v13/_search"]
    body = client.bodies[1]
    assert body is not None
    assert body["size"] == 0 and body["track_total_hits"] is True


# --------------------------------------------------------------------------- #
# odds and ends
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "index,expected",
    [
        ("catalog_fr_v13", "fr"),
        ("catalog_en_v8", "en"),
        ("catalog_es_v13", "es"),
        ("products", None),
    ],
)
def test_locale_is_inferred_from_the_index_name(index: str, expected: Optional[str]) -> None:
    assert infer_lang(index) == expected


def test_an_index_whose_name_says_nothing_needs_an_explicit_locale(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, distinct_ids=1)
    client = ReplayClient(search_response(total=1, with_path=1, taxonomy_tags={}, category_path={}))
    with pytest.raises(VerificationError, match="pass --lang"):
        verify(client, "products", manifest)
    assert check(verify(client, "products", manifest, lang="en"), "document_count")["status"] == (
        "pass"
    )
