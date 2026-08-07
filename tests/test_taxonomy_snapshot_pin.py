"""The taxonomy snapshot is pinned: a run that cannot get *it* stops.

Why this exists
---------------
``ensure_taxonomy`` used to treat "the snapshot is missing" as "download a fresh
one from upstream". Every category address in a catalog is a function of that
file, so the fallback made the pin mean "whatever upstream published today"
precisely on the run where it was needed held fixed — and it did so silently,
producing a complete, plausible, *unrecoverably* mislabelled catalog. Nothing
downstream can detect it afterwards: every product still gets a sensible-looking
``category_path``, just filed against a different taxonomy than the build it is
meant to be comparable to. The only place to catch it is before the run starts,
which is what these tests hold in place.

The two refusals under test, and the two ways each could rot
-------------------------------------------------------------
* **Missing snapshot, no opt-in** → refuse, naming the path and the expected
  digest. The failure mode this guards is not "the run errors": if the guard
  were deleted the run would *also* end at exit 2, because the freshly
  downloaded upstream file would then fail the digest check. So the assertion
  that actually bites is ``urlopen`` never being called and the file never
  appearing on disk — a "no exception raised"-shaped assertion could not tell
  the two apart.
* **A file that is not the pinned snapshot** → refuse, naming *both* digests.
  ``path.exists()`` waves through a stale, truncated or hand-edited file.

Each refusal is paired with a positive control that runs the *same* command with
the waiver flag and asserts exit 0 with a record written, so a guard that
refused unconditionally — or a harness broken for some unrelated reason — fails
here rather than reading as a working gate.

Everything drives the real ``extract.main()`` CLI. The guard lives on the path
between ``--taxonomy`` and ``load_taxonomy``; calling ``resolve_taxonomy``
directly would prove the function works while the CLI still walked past it.

Run with ``pytest tests/`` or directly:
``python tests/test_taxonomy_snapshot_pin.py``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from off_demo_extract import extract as extract_module  # noqa: E402
from off_demo_extract.extract import main  # noqa: E402
from off_demo_extract.taxonomy import (  # noqa: E402
    PINNED_TAXONOMY_SHA256,
    TaxonomySnapshotError,
    resolve_taxonomy,
)

PRICING_CONFIG = REPO_ROOT / "config" / "pricing_buckets.json"

# A miniature taxonomy. Its whole point here is that it is *not* the pinned
# snapshot: a hand-built fixture can never hash to the pin, which is what makes
# it the natural stand-in for "a file exists at the path but is the wrong one".
TAXONOMY = {
    "en:beverages": {"name": {"en": "Beverages"}, "parents": []},
    "en:hot-beverages": {"name": {"en": "Hot beverages"}, "parents": ["en:beverages"]},
    "en:teas": {"name": {"en": "Teas"}, "parents": ["en:hot-beverages"]},
}

# Carries a title, a description *and* an image: the extractor drops a record
# missing any of the three, and a positive control that produced no record for
# one of those reasons would be indistinguishable from a run the taxonomy guard
# had stopped.
PRODUCT = {
    "code": "3017620422003",
    "lang": "en",
    "product_name_en": "Earl Grey Tea Bags",
    "generic_name_en": "Black tea with bergamot, 20 bags",
    "categories_tags": ["en:beverages", "en:hot-beverages", "en:teas"],
    "images": {"front_en": {"rev": "7", "sizes": {"400": {"w": 400, "h": 400}}}},
}

TAXONOMY_BYTES = json.dumps(TAXONOMY).encode("utf-8")

# Computed here with hashlib directly rather than through the module's own
# ``taxonomy_sha256``: a test that hashed the file the same way the guard does
# would agree with a broken hasher instead of checking it.
FIXTURE_SHA256 = hashlib.sha256(TAXONOMY_BYTES).hexdigest()


class _Recorder:
    """Stands in for ``urllib.request.urlopen`` and records every call.

    The real function must never be reached from a test — a suite that fetched
    the live taxonomy would be slow, offline-hostile, and would stop testing the
    thing it claims to. The recorder also makes "did the run try to download?"
    an observable fact rather than an inference from the exit status.
    """

    def __init__(self, payload: bytes = TAXONOMY_BYTES) -> None:
        self.payload = payload
        self.urls: List[str] = []

    def __call__(self, url: str, *args: Any, **kwargs: Any) -> Any:
        self.urls.append(url)
        return _Response(self.payload)


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _Run:
    """One CLI invocation's inputs and outcome."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.taxonomy_path = tmp / "categories.json"
        self.output_path = tmp / "out.ndjson"
        self.rc = 0
        self.stderr = ""
        self.downloads: List[str] = []
        # Read eagerly, at the end of the run: the tests assert on the outcome
        # after the temporary directory has been cleaned up, and a lazy read
        # would come back empty there — which is what a refused run looks like.
        self.records: List[dict] = []

    def collect(self) -> None:
        if not self.output_path.exists():
            return
        self.records = [
            json.loads(line)
            for line in self.output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def _run(
    tmp: Path,
    *flags: str,
    write_snapshot: bool,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    urlopen: Any = None,
) -> _Run:
    """Drive the real CLI over one product, with the download stubbed out."""
    run = _Run(tmp)
    if write_snapshot:
        run.taxonomy_path.write_bytes(TAXONOMY_BYTES)

    recorder = _Recorder()
    monkeypatch.setattr(urllib.request, "urlopen", urlopen or recorder)

    input_path = tmp / "products.jsonl"
    input_path.write_text(json.dumps(PRODUCT) + "\n", encoding="utf-8")

    run.rc = main(
        [
            "--input", str(input_path),
            "--output", str(run.output_path),
            "--report", str(tmp / "report.json"),
            "--taxonomy", str(run.taxonomy_path),
            "--pricing-config", str(PRICING_CONFIG),
            "--progress-every", "0",
            "--progress-seconds", "0",
            "--yes",
            *flags,
        ]
    )
    run.stderr = capsys.readouterr().err
    run.downloads = recorder.urls
    run.collect()
    return run


def _tmpdir() -> Any:
    return tempfile.TemporaryDirectory(prefix="off-taxonomy-pin-test-")


# --------------------------------------------------------------------------
# Refusal 1: the snapshot is missing and nobody asked for a download.
# --------------------------------------------------------------------------


def test_a_missing_snapshot_stops_the_run(capsys, monkeypatch) -> None:
    """No snapshot and no ``--fetch-taxonomy`` is a failed run, not a fetch."""
    with _tmpdir() as tmp:
        run = _run(Path(tmp), write_snapshot=False, capsys=capsys, monkeypatch=monkeypatch)
    assert run.rc == 2, f"expected exit 2, got {run.rc}; stderr was: {run.stderr}"
    assert run.records == [], "a run that could not resolve its taxonomy still wrote a catalog"


def test_a_missing_snapshot_downloads_nothing(capsys, monkeypatch) -> None:
    """The load-bearing assertion: no request is made and no file appears.

    This is what separates a working guard from a deleted one. With the guard
    removed the run still exits 2 — the fetched upstream file fails the digest
    check — so the exit status alone proves nothing here.
    """
    with _tmpdir() as tmp:
        run = _run(Path(tmp), write_snapshot=False, capsys=capsys, monkeypatch=monkeypatch)
        landed = run.taxonomy_path.exists()
    assert run.downloads == [], f"the run fetched the taxonomy unasked: {run.downloads}"
    assert not landed, "the run wrote a taxonomy file it was never asked to fetch"


def test_the_missing_snapshot_refusal_names_the_path_and_the_pin(capsys, monkeypatch) -> None:
    """An operator has to be told *which* file, and which digest it must have."""
    with _tmpdir() as tmp:
        run = _run(Path(tmp), write_snapshot=False, capsys=capsys, monkeypatch=monkeypatch)
        expected_path = str(run.taxonomy_path)
    assert expected_path in run.stderr, run.stderr
    assert PINNED_TAXONOMY_SHA256 in run.stderr, run.stderr
    assert "--fetch-taxonomy" in run.stderr, run.stderr


def test_no_taxonomy_still_runs_with_nothing_on_disk(capsys, monkeypatch) -> None:
    """``--no-taxonomy`` is unaffected — the guard is not over-broad.

    The run asks for no taxonomy, so there is no snapshot for it to be wrong
    about. A guard placed before that check would break the one documented way
    to run without the file.
    """
    with _tmpdir() as tmp:
        run = _run(
            Path(tmp), "--no-taxonomy", "--no-require-category-path",
            write_snapshot=False, capsys=capsys, monkeypatch=monkeypatch,
        )
    assert run.rc == 0, f"--no-taxonomy exited {run.rc}: {run.stderr}"
    assert len(run.records) == 1, run.records
    assert run.records[0]["category_path"] == [], run.records[0]["category_path"]
    assert run.downloads == [], run.downloads


# --------------------------------------------------------------------------
# Refusal 2: a file is there, but it is not the pinned snapshot.
# --------------------------------------------------------------------------


def test_a_snapshot_that_is_not_the_pinned_one_stops_the_run(capsys, monkeypatch) -> None:
    """``path.exists()`` used to be the whole check; the digest is now too."""
    with _tmpdir() as tmp:
        run = _run(Path(tmp), write_snapshot=True, capsys=capsys, monkeypatch=monkeypatch)
    assert run.rc == 2, f"expected exit 2, got {run.rc}; stderr was: {run.stderr}"
    assert run.records == [], "a run built a catalog from an unpinned taxonomy"


def test_the_mismatch_refusal_names_both_digests(capsys, monkeypatch) -> None:
    """Naming only one digest leaves the reader unable to tell what they have."""
    with _tmpdir() as tmp:
        run = _run(Path(tmp), write_snapshot=True, capsys=capsys, monkeypatch=monkeypatch)
    assert PINNED_TAXONOMY_SHA256 in run.stderr, run.stderr
    assert FIXTURE_SHA256 in run.stderr, run.stderr
    assert FIXTURE_SHA256 != PINNED_TAXONOMY_SHA256, "the fixture must not be the pinned snapshot"


def test_the_same_run_completes_when_the_waiver_is_named(capsys, monkeypatch) -> None:
    """Positive control for refusal 2.

    Same command, same fixture, plus ``--allow-unpinned-taxonomy``: exit 0 and a
    record with a real ``category_path``. Without this, a guard that refused
    every run — or one refusing for an unrelated reason — would read as correct.
    """
    with _tmpdir() as tmp:
        run = _run(
            Path(tmp), "--allow-unpinned-taxonomy",
            write_snapshot=True, capsys=capsys, monkeypatch=monkeypatch,
        )
    assert run.rc == 0, f"the waived run exited {run.rc}: {run.stderr}"
    assert len(run.records) == 1, run.records
    assert run.records[0]["category_path"] == [
        "Beverages",
        "Beverages/Hot beverages",
        "Beverages/Hot beverages/Teas",
    ], run.records[0]["category_path"]


def test_a_matching_snapshot_completes_a_pinned_run(capsys, monkeypatch) -> None:
    """Positive control for the *pinned* branch, through the real CLI.

    The waiver control above proves a run completes with the check switched
    off — it never enters the comparison at all, so a digest branch that
    refused every file it looked at would leave it green. This is the path an
    actual build takes: pin enforced, file matches, run proceeds. The pin is
    moved to the fixture's digest rather than the fixture to the pin, because a
    hand-built taxonomy cannot be made to hash to the real snapshot's value.
    """
    monkeypatch.setattr(extract_module, "PINNED_TAXONOMY_SHA256", FIXTURE_SHA256)
    with _tmpdir() as tmp:
        run = _run(Path(tmp), write_snapshot=True, capsys=capsys, monkeypatch=monkeypatch)
    assert run.rc == 0, f"a run against the pinned snapshot exited {run.rc}: {run.stderr}"
    assert run.downloads == [], run.downloads
    assert len(run.records) == 1, run.records
    assert run.records[0]["category_path"] == [
        "Beverages",
        "Beverages/Hot beverages",
        "Beverages/Hot beverages/Teas",
    ], run.records[0]["category_path"]


# --------------------------------------------------------------------------
# The opt-in download, and what it does *not* opt out of.
# --------------------------------------------------------------------------


def test_the_opt_in_flag_downloads_the_snapshot(capsys, monkeypatch) -> None:
    """``--fetch-taxonomy`` restores the old behaviour, on request only."""
    with _tmpdir() as tmp:
        run = _run(
            Path(tmp), "--fetch-taxonomy", "--allow-unpinned-taxonomy",
            write_snapshot=False, capsys=capsys, monkeypatch=monkeypatch,
        )
        landed = run.taxonomy_path.read_bytes()
    assert run.downloads == [
        "https://static.openfoodfacts.org/data/taxonomies/categories.json"
    ], run.downloads
    assert landed == TAXONOMY_BYTES
    assert run.rc == 0, f"the fetching run exited {run.rc}: {run.stderr}"
    assert len(run.records) == 1, run.records


def test_a_failed_download_is_the_same_named_error(capsys, monkeypatch) -> None:
    """An opt-in fetch that cannot complete exits 2, not with a traceback.

    The caller has one exception type to catch either way; a bare ``URLError``
    escaping ``main`` would end the run with a stack trace instead of a sentence
    about the taxonomy.
    """

    def _refuse(url: str, *args: Any, **kwargs: Any) -> Any:
        raise urllib.error.URLError("connection refused")

    with _tmpdir() as tmp:
        run = _run(
            Path(tmp), "--fetch-taxonomy",
            write_snapshot=False, capsys=capsys, monkeypatch=monkeypatch, urlopen=_refuse,
        )
    assert run.rc == 2, f"expected exit 2, got {run.rc}; stderr was: {run.stderr}"
    assert "connection refused" in run.stderr, run.stderr
    assert run.records == []


def test_the_opt_in_flag_does_not_opt_out_of_the_pin(capsys, monkeypatch) -> None:
    """A downloaded file is checked against the pin like any other.

    Otherwise ``--fetch-taxonomy`` would be a way to build against whatever
    upstream serves today while still reading as a pinned build — the original
    defect, one flag further along.
    """
    with _tmpdir() as tmp:
        run = _run(
            Path(tmp), "--fetch-taxonomy",
            write_snapshot=False, capsys=capsys, monkeypatch=monkeypatch,
        )
    assert run.downloads, "the opt-in did not fetch"
    assert run.rc == 2, f"expected exit 2, got {run.rc}; stderr was: {run.stderr}"
    assert PINNED_TAXONOMY_SHA256 in run.stderr, run.stderr
    assert FIXTURE_SHA256 in run.stderr, run.stderr
    assert run.records == []


# --------------------------------------------------------------------------
# The comparison itself, and the constant it compares against.
# --------------------------------------------------------------------------


def test_resolve_taxonomy_accepts_the_digest_it_was_given() -> None:
    """The check compares; it does not simply always refuse."""
    with _tmpdir() as tmp:
        path = Path(tmp) / "categories.json"
        path.write_bytes(TAXONOMY_BYTES)
        assert resolve_taxonomy(path, expected_sha256=FIXTURE_SHA256) == path


def test_resolve_taxonomy_refuses_one_flipped_byte() -> None:
    """The digest, not the size or the shape, is what is being compared."""
    with _tmpdir() as tmp:
        path = Path(tmp) / "categories.json"
        path.write_bytes(TAXONOMY_BYTES.replace(b"Teas", b"teas"))
        with pytest.raises(TaxonomySnapshotError) as caught:
            resolve_taxonomy(path, expected_sha256=FIXTURE_SHA256)
        assert FIXTURE_SHA256 in str(caught.value)


def test_the_pin_is_the_digest_every_build_record_carries() -> None:
    """The constant must be the recorded pin, not a plausible-looking typo.

    ``PINNED_TAXONOMY_SHA256`` is a 64-character literal that nothing else would
    catch if it were mistyped: a wrong value simply refuses every run, and a
    right-looking wrong value would refuse the *correct* snapshot. The build
    records under ``builds/`` are where that digest was actually observed, so
    they are what it is checked against.
    """
    recorded = set()
    for name in sorted(glob.glob(str(REPO_ROOT / "builds" / "*" / "build_manifest.json"))):
        recorded.add(json.loads(Path(name).read_text(encoding="utf-8"))["taxonomy"]["sha256"])
    for name in sorted(glob.glob(str(REPO_ROOT / "builds" / "*" / "index_verify_*.json"))):
        for check in json.loads(Path(name).read_text(encoding="utf-8"))["checks"]:
            if "pinned_taxonomy_sha256" in check:
                recorded.add(check["pinned_taxonomy_sha256"])

    assert recorded, "no build record carried a taxonomy digest to check the pin against"
    assert recorded == {PINNED_TAXONOMY_SHA256}, (
        "PINNED_TAXONOMY_SHA256 is not the digest the build records pin: "
        f"constant {PINNED_TAXONOMY_SHA256}, records {sorted(recorded)}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
