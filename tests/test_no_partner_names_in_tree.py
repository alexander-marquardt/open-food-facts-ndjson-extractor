"""Guard: no customer or partner organisation is named in this repository.

This repository is public. Naming an organisation we work with attaches a
commercial relationship — and, worse, their internal data conventions — to a
public artifact. That is theirs to publish, not ours. The sibling repositories
treat this as a hard rule; this test is the gate that enforces it here, and it
runs in CI (``.github/workflows/tests.yml`` runs the whole suite) as well as
locally, which is why it lives in ``tests/`` rather than as a separate workflow
step.

**The ban list is stored as SHA-256 digests, not as literal strings**, so that
writing the guard does not itself put the names it bans into the public tree —
which would defeat the entire point. The digests are irreversible: a reader of
this file learns that a name is banned, not which one.

To add a name to the ban list, append the output of::

    python -c 'import hashlib,sys;print(hashlib.sha256(sys.argv[1].strip().lower().encode()).hexdigest())' 'THE NAME'

Known limits, stated plainly so nobody mistakes this for more than it is:

* It matches whole alphabetic tokens (including camel-case segments), so a name
  split across punctuation-free glue like ``acmeretail`` is not caught.
* It matches single tokens only; a multi-word organisation name is caught only
  if one of its words is distinctive enough to be listed on its own.
* It scans tracked, text-decodable files. Binary blobs are skipped.
* It fixes ``HEAD`` only. These names remain in this repository's git history;
  removing them from there is a separate, explicit decision (see issue #23).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# SHA-256 of each banned name, lowercased and stripped. See the module docstring
# for why these are digests and how to add one.
BANNED_DIGESTS = frozenset(
    {
        "82f609daa5a34f99b8ff991d41962aa511c7e1d2c20887760dbf9f874e1f2785",
    }
)

_RUN_RE = re.compile(r"[A-Za-z]+")
# Camel-case segments within a run: "FooBarAPI" -> Foo, Bar, API.
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")


def _tokens(line: str) -> Iterable[str]:
    """Every alphabetic run, plus its camel-case segments, lowercased.

    Both are needed: ``a-name-here`` only yields the run, while ``ANameHere``
    only yields segments. Emitting just one of the two leaves a hole a name can
    slip through — the positive-control test pins exactly this.
    """
    for run in _RUN_RE.finditer(line):
        word = run.group(0)
        yield word.lower()
        for segment in _CAMEL_RE.finditer(word):
            yield segment.group(0).lower()


def _scan_lines(lines: Sequence[str], digests: frozenset) -> List[int]:
    """1-based line numbers whose tokens hash into ``digests``."""
    hits = []
    for number, line in enumerate(lines, start=1):
        for token in _tokens(line):
            if hashlib.sha256(token.encode()).hexdigest() in digests:
                hits.append(number)
                break
    return hits


def _tracked_text_files() -> List[Path]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "cannot enumerate tracked files, so this guard cannot run: "
            f"git ls-files exited {result.returncode}: "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    names = [n for n in result.stdout.decode("utf-8").split("\0") if n]
    assert names, "git reported no tracked files; the guard would be vacuous"
    return [REPO_ROOT / name for name in names]


def _decode(path: Path) -> str | None:
    """File text, or ``None`` when the file is binary / undecodable."""
    try:
        raw = path.read_bytes()
    except OSError:  # pragma: no cover - a tracked path that cannot be read
        return None
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def test_scanner_is_not_inert() -> None:
    """Positive control: the machinery must fire on a planted token.

    Without this, a typo in the token regex or an empty ban list would leave
    :func:`test_tracked_content_names_no_partner` passing on any tree at all —
    a guard that reads as protection while protecting nothing. The sentinel is
    a harmless word, so proving the scanner bites costs us no exposure.
    """
    sentinel = "sentineltoken"
    digests = frozenset({hashlib.sha256(sentinel.encode()).hexdigest()})

    assert _scan_lines(["nothing to see here"], digests) == []
    assert _scan_lines(["a SentinelToken in camel case"], digests) == [1]
    assert _scan_lines(["clean", "the-sentineltoken-here", "clean"], digests) == [2]
    assert _scan_lines(["prefixed_sentineltoken_suffixed"], digests) == [1]


def test_ban_list_is_populated() -> None:
    """An empty ban list would make the tree scan trivially green."""
    assert BANNED_DIGESTS, "the ban list is empty; the tree scan would be vacuous"
    for digest in BANNED_DIGESTS:
        assert re.fullmatch(r"[0-9a-f]{64}", digest), (
            f"ban-list entry is not a SHA-256 hex digest: {digest!r}"
        )


def test_tracked_content_names_no_partner() -> None:
    """No tracked, text-decodable file names a banned organisation."""
    offences: List[Tuple[str, int]] = []
    scanned = 0

    for path in _tracked_text_files():
        text = _decode(path)
        if text is None:
            continue
        scanned += 1
        relative = path.relative_to(REPO_ROOT).as_posix()
        for number in _scan_lines(text.splitlines(), BANNED_DIGESTS):
            offences.append((relative, number))

    assert scanned, "no text files were scanned; the guard would be vacuous"
    assert not offences, (
        "a banned customer/partner name appears in tracked content at "
        + ", ".join(f"{name}:{number}" for name, number in offences)
        + " — describe the shape ('the shape retail catalogs typically expose'), "
        "not the organisation. This repository is public. See issue #23. "
        "(The offending text is deliberately not echoed here.)"
    )
