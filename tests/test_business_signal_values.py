"""Business-signal derivation: popularity is observed, margin is modelled.

Both fields were a uniform random draw seeded on the GTIN until
elastic/prism#5027. These tests fail on that behaviour: they pin that
``popularity`` is a monotone function of the dump's real ``unique_scans_n`` and
nothing else, and that ``margin`` depends on the product's category and label
tags rather than on its barcode.

They also pin the exact numbers the PRISM customization repo's migration script
(``scripts/business_signal_values.py`` in elastic/prism-open-food-facts)
produces, because a catalogue re-extracted here has to agree with a catalogue
migrated there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from off_demo_extract.extract import (  # noqa: E402
    BUCKET_BASE_MARGIN_PCT,
    LABEL_UPLIFT_CAP,
    POPULARITY_MAX,
    derive_margin,
    derive_popularity,
)


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
