"""Business-signal derivation: popularity is observed, margin is modelled.

Both fields were a uniform random draw seeded on the GTIN until the draw was
diagnosed and replaced. These tests pin that ``popularity`` is a monotone
function of the dump's real ``unique_scans_n`` and nothing else, and that
``margin`` depends on the product's category and label tags rather than on its
barcode.

The last section runs the removed draw side by side with its replacement, so
the claim that the old values carried no information is demonstrated by
executing both, not asserted in prose.

They also pin the exact numbers these derivations produce, digit for digit.
That matters because an already-indexed catalog is migrated in place by a
separate script rather than re-extracted from the dump: the two paths have to
land on identical values, or the same product carries one margin in a
freshly built index and a different one in a migrated index of the same data.
Changing a constant here without changing it there is the failure this pins.
"""

from __future__ import annotations

import pytest

from off_demo_extract.extract import (
    BUCKET_BASE_MARGIN_PCT,
    LABEL_UPLIFT_CAP,
    POPULARITY_MAX,
    derive_margin,
    derive_popularity,
)
from off_demo_extract.pricing import _seeded_rng


@pytest.mark.parametrize(
    ("scans", "expected"),
    [(None, 0), (0, 0), (1, 693), (10, 2398), (26, 3296), (1000, 6909)],
)
def test_popularity_is_the_documented_log_mapping(scans, expected):
    assert derive_popularity(scans) == expected


def test_popularity_is_monotone_and_capped():
    counts = [0, 1, 2, 5, 25, 400, 9_000, 250_000]
    values = [derive_popularity(c) for c in counts]
    assert values == sorted(values)
    assert derive_popularity(10**9) == POPULARITY_MAX


def test_margin_takes_the_bucket_rate_and_the_real_label_uplift():
    plain, plain_detail = derive_margin("snacks_sweets", [])
    assert plain == BUCKET_BASE_MARGIN_PCT["snacks_sweets"]
    assert "bucket=snacks_sweets" in plain_detail

    lifted, lifted_detail = derive_margin("snacks_sweets", ["en:no-gluten"])
    assert lifted > plain
    assert "no-gluten" in lifted_detail


def test_unrecognised_labels_do_not_move_the_margin():
    with_noise, detail = derive_margin("dairy", ["en:vegan", "en:green-dot"])
    assert with_noise == BUCKET_BASE_MARGIN_PCT["dairy"]
    assert "vegan" not in detail


def test_compounded_uplift_is_capped():
    margin, _ = derive_margin("bakery", ["en:organic", "en:fair-trade", "en:no-gluten"])
    assert margin == round(BUCKET_BASE_MARGIN_PCT["bakery"] * LABEL_UPLIFT_CAP)


def test_unknown_bucket_falls_back_to_the_default_rate():
    for bucket in ("", "a_bucket_the_pricing_model_never_emits"):
        margin, detail = derive_margin(bucket, None)
        assert margin == BUCKET_BASE_MARGIN_PCT["default"]
        assert "bucket=default" in detail


# --------------------------------------------------------------------------- #
# The replaced draw, kept executable next to its replacement
# --------------------------------------------------------------------------- #
#
# These tests cannot be run against the pre-change source to show them failing:
# ``derive_popularity`` / ``derive_margin`` did not exist there, so a checkout of
# the old revision fails to import rather than fails an assertion, which is no
# evidence at all. The comparison is therefore made HERE, with both paths
# runnable in the same process — the draw that was removed on the left, the
# derivation that replaced it on the right. Every assertion below on the legacy
# side is a property the shipped derivation does not have, and vice versa.


def _legacy_margin_and_popularity(gtin: str) -> tuple[int, int]:
    """``generate_margin_and_popularity`` exactly as it stood before this change.

    Reproduced rather than imported, because the function is deleted. It draws
    from a generator seeded on the barcode and nothing else, so neither number
    can carry any information about the product it is attached to.
    """
    rng = _seeded_rng(gtin, "margin_popularity")
    return rng.randint(0, 200), rng.randint(0, 10000)


_BARCODES = [f"{n:013d}" for n in range(2000)]


def test_the_draw_had_the_uniform_signature_the_catalogues_were_measured_to_have():
    """The fill was identified from its mean landing on the midpoint of the range.

    Reproduced here so the diagnosis is checkable and not taken on trust: over a
    barcode sample the draw's means sit on the midpoints of 0..200 and 0..10000,
    which is what was measured on all three published catalogues independently.
    """
    margins = [_legacy_margin_and_popularity(g)[0] for g in _BARCODES]
    populars = [_legacy_margin_and_popularity(g)[1] for g in _BARCODES]
    assert abs(sum(margins) / len(margins) - 100) < 3
    assert abs(sum(populars) / len(populars) - 5000) < 150

    # The derivation has no such midpoint: margin is a category rate, so its
    # values cluster low in the same 0..200 envelope rather than filling it.
    derived = [derive_margin(bucket, [])[0] for bucket in BUCKET_BASE_MARGIN_PCT]
    assert max(derived) < 100


def test_the_draw_moved_with_the_barcode_where_the_model_moves_with_the_product():
    """Two products in the same category with the same claims must price the same.

    The draw gave them unrelated margins — which is precisely why raising the
    margin weight in the demo promoted an arbitrary set of products and the
    merchandiser could point at nothing to explain it.
    """
    drawn = {_legacy_margin_and_popularity(g)[0] for g in _BARCODES}
    assert len(drawn) > 100

    modelled = {derive_margin("snacks_sweets", ["en:no-gluten"])[0] for _ in _BARCODES}
    assert modelled == {42}


def test_the_draw_contradicted_the_scan_order_the_derivation_reproduces():
    """popularity now orders products the way real scanners ordered them.

    Same five barcodes, ascending in their real Open Food Facts scan count. The
    derivation returns them in that order; the draw does not, because it never
    read the count.
    """
    sample = [
        ("0000000000017", 0),
        ("0000000000024", 3),
        ("0000000000031", 40),
        ("0000000000048", 900),
        ("0000000000055", 12_000),
    ]
    derived = [derive_popularity(scans) for _, scans in sample]
    assert derived == sorted(derived)
    assert derived[0] == 0
    assert derived[-1] < POPULARITY_MAX

    drawn = [_legacy_margin_and_popularity(gtin)[1] for gtin, _ in sample]
    assert drawn != sorted(drawn)
    # Under the draw the never-scanned product outranked a scanned one.
    assert drawn[0] > drawn[1]
