"""Guard: this public repository names no partner, and cites no repo you cannot open.

This repository is public. Two different things must therefore stay out of it,
and this file gates both over the same tracked-file scan — one walk of the tree,
two questions asked of each line — because a second mechanism in a second file is
a second thing to remember to run.

Part 1 — no customer or partner organisation is named
-----------------------------------------------------
Naming an organisation we work with attaches a commercial relationship — and,
worse, their internal data conventions — to a public artifact. That is theirs to
publish, not ours. The sibling repositories treat this as a hard rule; this test
is the gate that enforces it here, and it runs in CI
(``.github/workflows/tests.yml`` runs the whole suite) as well as locally, which
is why it lives in ``tests/`` rather than as a separate workflow step.

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

Part 2 — no reference to a repository a reader cannot open
----------------------------------------------------------
A rationale whose only citation is a link that 404s is not much of a rationale.
Comments here used to justify real design decisions by pointing at issues in a
non-public sibling project; GitHub renders those as links, and an outside reader
gets a 404 and no context at all. The fix is to inline the reason so the comment
stands on its own — a cross-repo issue number carries no meaning to someone who
cannot open it anyway. This part of the guard keeps the references from coming
back (see issue #26).

These are **patterns, not secrets** — the project is ours, not a partner's — so
unlike part 1 they are matched literally rather than by digest, which lets the
failure message name the offending text and point straight at it.

The pattern fragments below are **assembled from pieces at import time** for one
reason only: written whole, this file's own source would match them, and a guard
that always fails on itself is a guard someone deletes. Excluding this file from
the scan instead would leave a hole exactly where a reference is most likely to
be re-added by someone editing the guard. There is no attempt at concealment; the
paragraph you are reading says what is meant.

Known limits, again stated plainly:

* It lists the specific non-public projects known to have been cited here. It is
  deliberately *not* a general "internal reference" matcher: banning every
  ``owner/repo#123`` shape would also ban ``openfoodfacts/openfoodfacts-server``
  citations, which a reader **can** open and which this project should keep
  making.
* It says nothing about the internal project's *name* appearing on its own —
  ``PRISM_ELASTICSEARCH_URL`` is an environment variable this repository's
  scripts genuinely read, and renaming it would break their callers. Whether the
  bare project name belongs in a public repository at all is a separate
  question, tracked separately.
* Like part 1, it fixes ``HEAD`` only; the references remain in git history.
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

# Non-public repository references. Assembled from fragments so that this file
# does not match its own source — see the module docstring for why that matters
# more than it looks like it should.
_ORG = "elast" + "ic"
_PROJECT = "pri" + "sm"

NON_PUBLIC_REPO_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    # ``<org>/<project>``, with or without a repo suffix or an issue number:
    # catches both the issue citations and the sibling repository's own path.
    (
        f"{_ORG}/{_PROJECT}…",
        re.compile(rf"\b{_ORG}/{_PROJECT}[A-Za-z0-9._-]*(?:#[0-9]+)?", re.IGNORECASE),
    ),
    # The sibling repository's slug written without its organisation.
    (
        f"{_PROJECT}-open-food-facts",
        re.compile(rf"\b{_PROJECT}-open-food-facts\b", re.IGNORECASE),
    ),
    # A cross-repo issue citation with the organisation left off.
    (f"{_PROJECT}#N", re.compile(rf"\b{_PROJECT}#[0-9]+", re.IGNORECASE)),
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


def _scan_repo_references(lines: Sequence[str]) -> List[Tuple[int, str]]:
    """``(1-based line number, matched text)`` for every non-public repo reference.

    The patterns overlap on purpose — the org-qualified one and the bare-issue one
    both fire on ``<org>/<project>#5027`` — so a match wholly inside another match
    on the same line is folded away. Reporting both would make one reference look
    like two and point the reader at the middle of the string.
    """
    hits: List[Tuple[int, str]] = []
    for number, line in enumerate(lines, start=1):
        spans = [
            match.span()
            for _label, pattern in NON_PUBLIC_REPO_PATTERNS
            for match in pattern.finditer(line)
        ]
        outermost = [
            (start, end)
            for start, end in spans
            if not any(
                (other_start, other_end) != (start, end)
                and other_start <= start
                and end <= other_end
                for other_start, other_end in spans
            )
        ]
        for start, end in sorted(set(outermost)):
            hits.append((number, line[start:end]))
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


def test_repo_reference_scanner_is_not_inert() -> None:
    """Positive control: the matcher must fire on a reintroduced reference.

    The digest guard's own first draft passed on a real hole (issue #23), which
    is the reason every scanner in this file has to demonstrate that it bites
    before its green result means anything. Each shape below is one that was
    actually in the tree, or one line's worth of typing away from it.
    """
    org, project = _ORG, _PROJECT

    caught = [
        f"# (see {org}/{project}#5027).",
        f"the migration script in {org}/{project}-open-food-facts",
        f"identified in {project}#5027, reproduced here",
        f"Fixed by {org.upper()}/{project.upper()}#12",
    ]
    for line in caught:
        assert _scan_repo_references([line]), f"scanner missed: {line!r}"

    # Just as important: it must not fire on citations a reader *can* open, or
    # this guard becomes something contributors route around.
    allowed = [
        "``taxonomies/food/categories.txt`` in ``openfoodfacts/openfoodfacts-server``",
        "See openfoodfacts/openfoodfacts-server#8123 for the rename",
        "Follow-up from #9 / #18.",
        "More context: https://alexmarquardt.com/elastic/ecommerce-demo-data/",
        f"{org.upper()}_ELASTICSEARCH_URL and {org.upper()}_ELASTICSEARCH_API_KEY",
        f"the {project.upper()} field map splits paths on PATH_SEPARATOR",
    ]
    for line in allowed:
        assert not _scan_repo_references([line]), f"scanner false-positived on: {line!r}"

    assert _scan_repo_references(["clean", f"{org}/{project}#1", "clean"]) == [
        (2, f"{org}/{project}#1")
    ]


def test_repo_reference_patterns_do_not_match_this_file() -> None:
    """The guard must be able to describe what it bans without tripping on itself.

    If this ever fails, someone wrote a pattern whole instead of assembling it —
    at which point the tree scan below fails permanently and the temptation is to
    exclude this file from the scan, which is the one hole worth not having.
    """
    own_source = Path(__file__).read_text(encoding="utf-8")
    hits = _scan_repo_references(own_source.splitlines())
    assert not hits, (
        "the guard's own source matches its patterns at lines "
        f"{sorted({n for n, _ in hits})}; assemble the pattern from fragments "
        "rather than excluding this file from the scan"
    )


def test_tracked_content_cites_no_non_public_repo() -> None:
    """No tracked, text-decodable file points at a repository a reader cannot open."""
    offences: List[str] = []
    scanned = 0

    for path in _tracked_text_files():
        text = _decode(path)
        if text is None:
            continue
        scanned += 1
        relative = path.relative_to(REPO_ROOT).as_posix()
        for number, matched in _scan_repo_references(text.splitlines()):
            offences.append(f"{relative}:{number}: {matched}")

    assert scanned, "no text files were scanned; the guard would be vacuous"
    assert not offences, (
        "tracked content cites a repository an outside reader cannot open:\n  "
        + "\n  ".join(offences)
        + "\nThis repository is public and the link 404s for everyone else. Inline "
        "the reason instead ('to match the shape the consuming search engine "
        "expects'), so the comment stands on its own. See issue #26."
    )


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
