"""Tests for ``scripts/tolerance.py`` — is there exactly one tolerance rule?

Three scripts used to carry their own ``Tolerance`` with the same two-flag
shape, the same ``permitted()`` / ``describe()`` pair and the same
floor-never-round rule (#41). The copies were byte-identical in behaviour, which
is the whole hazard: a change to the rule could land in two of them and miss the
third, and all three would still run and still print a report, so nothing would
say so. Only ``verify_catalog.py`` had a test on the floor edge.

Deduplicating once does not stay deduplicated, so the durable half of this file
is not the edge test but the two structural ones:

* :func:`test_no_script_carries_a_second_copy_of_the_rule` walks **every** file
  in ``scripts/`` — found by glob, not by a list somebody has to remember to
  extend — and refuses a second implementation anywhere in the directory. A
  fourth script that hand-rolls the rule reds it on the day it is written.
* :func:`test_each_caller_builds_the_shared_class` refuses the other direction:
  a caller that stops using the shared class, or subclasses it to override
  ``permitted``/``describe``, reds too.

**Why the structural pair cannot pass vacuously.** A scanner that looks for
something and finds nothing is indistinguishable from a scanner that cannot see.
So the same three detectors are pointed at ``scripts/tolerance.py`` first, in
:func:`test_the_detector_fires_on_the_file_that_is_allowed_to_hold_the_rule`,
and each one has to fire there: the file defines a class named ``Tolerance``, it
defines both method names, and it floors a product. If a refactor ever silences
the detectors — renames the methods, stops flooring — that test goes red *before*
the sweep tests can go quietly green on an empty search. The glob is checked for
the three known callers for the same reason: a glob that matched nothing would
otherwise sweep nothing and pass.

The per-caller strings pinned below are the ones the three scripts printed at
355e022, the commit before the rule was hoisted, captured by running the old
classes rather than by reading them. #41's third acceptance criterion is that no
command line and no printed description changes, and this is what holds it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SHARED = SCRIPTS / "tolerance.py"

sys.path.insert(0, str(SCRIPTS))

import inject_category_path  # noqa: E402
import reindex_v7_to_v8  # noqa: E402
import tolerance as shared_tolerance  # noqa: E402
import verify_catalog  # noqa: E402

# The three scripts #41 names. Used only to prove the glob below is not empty —
# the sweep itself must cover scripts nobody has written yet, so it globs.
KNOWN_CALLERS = {
    "inject_category_path.py",
    "reindex_v7_to_v8.py",
    "verify_catalog.py",
}


# --------------------------------------------------------------------------- #
# the detectors
# --------------------------------------------------------------------------- #
def rule_markers(path: Path) -> List[str]:
    """Every sign that ``path`` implements the tolerance rule itself.

    Three independent shapes, so that hiding from one is not enough:

    * a class named after the rule,
    * either of the two method names the rule is made of,
    * flooring a product, which is the rule's arithmetic and nothing else's.

    The last one is deliberately narrower than "calls ``math.floor``":
    ``src/off_demo_extract/pricing.py`` floors a price, which is a different
    thing and must not be swept up.
    """
    found: List[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and "tolerance" in node.name.lower():
            found.append(f"class {node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in {"permitted", "describe"}:
                found.append(f"def {node.name}")
        elif isinstance(node, ast.Call):
            callee = node.func
            floors = (
                isinstance(callee, ast.Attribute)
                and callee.attr == "floor"
                or isinstance(callee, ast.Name)
                and callee.id == "floor"
            )
            product = (
                len(node.args) == 1
                and isinstance(node.args[0], ast.BinOp)
                and isinstance(node.args[0].op, ast.Mult)
            )
            if floors and product:
                found.append("floor(a * b)")
    return sorted(found)


def swept_files() -> List[Path]:
    """Every Python file that is not allowed to hold a copy of the rule."""
    return sorted(
        path
        for path in list(SCRIPTS.glob("*.py")) + list((REPO_ROOT / "src").rglob("*.py"))
        if path != SHARED
    )


# --------------------------------------------------------------------------- #
# the rule has exactly one home
# --------------------------------------------------------------------------- #
def test_the_detector_fires_on_the_file_that_is_allowed_to_hold_the_rule() -> None:
    """The anti-vacuity check: prove the detectors can see what they look for.

    Without this, a sweep that finds nothing proves nothing — a renamed method
    or a rewritten arithmetic would silence every detector and turn the two
    sweeps below into tests that cannot fail.
    """
    markers = rule_markers(SHARED)

    assert "class Tolerance" in markers
    assert "def permitted" in markers
    assert "def describe" in markers
    assert "floor(a * b)" in markers


def test_the_sweep_actually_reaches_the_three_scripts_that_had_a_copy() -> None:
    """A glob that matched nothing would sweep nothing and pass."""
    swept = {path.name for path in swept_files()}

    assert KNOWN_CALLERS <= swept
    assert "tolerance.py" not in swept


def test_no_script_carries_a_second_copy_of_the_rule() -> None:
    """A fourth copy, anywhere under ``scripts/`` or ``src/``, is a failure.

    This is the test #41 exists for. It is written against a glob rather than a
    list of three names so that a script written next month is covered without
    anybody remembering to add it here.
    """
    offenders = {
        path.relative_to(REPO_ROOT).as_posix(): rule_markers(path)
        for path in swept_files()
        if rule_markers(path)
    }

    assert offenders == {}, (
        "the floor-never-round tolerance rule has more than one home: "
        f"{offenders}. Import Tolerance from scripts/tolerance.py instead of "
        "restating the rule."
    )


@pytest.mark.parametrize(
    "module, factory_name",
    [
        (inject_category_path, "missing_tolerance"),
        (reindex_v7_to_v8, "missing_tolerance"),
        (verify_catalog, "values_tolerance"),
    ],
)
def test_each_caller_builds_the_shared_class(module: object, factory_name: str) -> None:
    """The other direction: a caller that stops using the shared rule reds.

    ``type(...) is`` rather than ``isinstance``: a subclass that overrode
    ``permitted`` or ``describe`` would satisfy ``isinstance`` while being
    exactly the divergence this issue is about.
    """
    factory = getattr(module, factory_name)
    built = factory()

    assert type(built) is shared_tolerance.Tolerance
    assert getattr(module, "Tolerance") is shared_tolerance.Tolerance
    assert type(built).permitted is shared_tolerance.Tolerance.permitted
    assert type(built).describe is shared_tolerance.Tolerance.describe


# --------------------------------------------------------------------------- #
# the rule itself, at the edge it was argued over
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fraction, counted, permitted",
    [
        # The two the three pull requests argued about by name.
        (0.5, 5, 2),  # not 3
        (0.29, 10, 2),  # not 3
        # Rounding would differ from flooring at every one of these.
        (0.5, 7, 3),
        (0.6, 5, 3),
        (0.9999, 10, 9),
        (0.01, 99, 0),
        # And must not differ where there is nothing to round.
        (0.0, 100, 0),
        (1.0, 100, 100),
        (0.5, 0, 0),
    ],
)
def test_a_fraction_is_floored_and_never_rounded(
    fraction: float, counted: int, permitted: int
) -> None:
    tolerance = shared_tolerance.Tolerance("--allow", "--allow-fraction", None, fraction)

    assert tolerance.permitted(counted) == permitted


def test_a_whole_number_wins_over_a_fraction() -> None:
    """Mutually exclusive on every command line; stated here rather than assumed."""
    tolerance = shared_tolerance.Tolerance("--allow", "--allow-fraction", 3, 0.9)

    assert tolerance.permitted(1000) == 3


def test_naming_nothing_is_zero_and_says_so() -> None:
    tolerance = shared_tolerance.Tolerance("--allow", "--allow-fraction")

    assert tolerance.permitted(1_000_000) == 0
    assert tolerance.describe(1_000_000) == "0 (default: zero tolerance)"


def test_describe_records_what_was_permitted_not_what_was_requested() -> None:
    """0.29 of 10 was *requested*; 2 is what the exit status was decided on."""
    tolerance = shared_tolerance.Tolerance("--allow", "--allow-fraction", None, 0.29)

    assert tolerance.describe(10) == "--allow-fraction 0.29 (2 of 10)"


@pytest.mark.parametrize(
    "allow, fraction, problem",
    [
        (None, None, None),
        (0, None, None),
        (5, None, None),
        (-1, None, "--allow must not be negative"),
        (None, 0.0, None),
        (None, 1.0, None),
        (None, 1.5, "--allow-fraction must be between 0 and 1"),
        (None, -0.1, "--allow-fraction must be between 0 and 1"),
    ],
)
def test_an_unusable_tolerance_is_named_with_the_flag_that_carried_it(
    allow: Optional[int], fraction: Optional[float], problem: Optional[str]
) -> None:
    tolerance = shared_tolerance.Tolerance("--allow", "--allow-fraction", allow, fraction)

    assert tolerance.problem() == problem


# --------------------------------------------------------------------------- #
# no command line and no printed description changed
# --------------------------------------------------------------------------- #
# (factory, arguments, denominator, exactly what 355e022 printed)
PRINTED_AT_355e022: List[Tuple[str, Tuple[Optional[int], Optional[float]], int, str]] = [
    ("inject", (None, None), 108380, "0 (default: zero tolerance)"),
    ("inject", (7, None), 108380, "--allow-missing 7"),
    ("inject", (0, None), 10, "--allow-missing 0"),
    ("inject", (None, 0.29), 10, "--allow-missing-fraction 0.29 (2 of 10)"),
    ("inject", (None, 0.5), 7, "--allow-missing-fraction 0.5 (3 of 7)"),
    ("inject", (None, 0.01), 108380, "--allow-missing-fraction 0.01 (1,083 of 108,380)"),
    ("inject", (None, 1.0), 5, "--allow-missing-fraction 1 (5 of 5)"),
    ("reindex", (None, None), 108380, "0 (default: zero tolerance)"),
    ("reindex", (7, None), 108380, "--allow-missing 7"),
    ("reindex", (None, 0.29), 10, "--allow-missing-fraction 0.29 (2 of 10)"),
    ("reindex", (None, 0.01), 108380, "--allow-missing-fraction 0.01 (1,083 of 108,380)"),
    ("catalog", (None, None), 3977, "0 (default: zero tolerance)"),
    ("catalog", (7, None), 3977, "--allow-values-outside-snapshot 7"),
    (
        "catalog",
        (None, 0.5),
        7,
        "--allow-values-outside-snapshot-fraction 0.5 (3 of 7)",
    ),
    (
        "catalog",
        (None, 0.29),
        10,
        "--allow-values-outside-snapshot-fraction 0.29 (2 of 10)",
    ),
    (
        "catalog",
        (None, 0.01),
        108380,
        "--allow-values-outside-snapshot-fraction 0.01 (1,083 of 108,380)",
    ),
]

FACTORY_NAMES = {
    "inject": (inject_category_path, "missing_tolerance"),
    "reindex": (reindex_v7_to_v8, "missing_tolerance"),
    "catalog": (verify_catalog, "values_tolerance"),
}


def factory(caller: str):
    """Resolved on call, not at import.

    A module-level ``getattr`` would turn a caller that does not have the
    factory into a collection error, and a collection error takes the whole
    file down — including the sweeps, which are the tests that would have named
    the actual problem.
    """
    module, name = FACTORY_NAMES[caller]
    return getattr(module, name)


@pytest.mark.parametrize("caller, arguments, counted, printed", PRINTED_AT_355e022)
def test_each_caller_still_prints_exactly_what_it_printed_before_the_hoist(
    caller: str,
    arguments: Tuple[Optional[int], Optional[float]],
    counted: int,
    printed: str,
) -> None:
    assert factory(caller)(*arguments).describe(counted) == printed


# (script, whole-number flag, fraction flag)
CALLER_FLAGS = [
    ("inject_category_path.py", "--allow-missing", "--allow-missing-fraction"),
    ("reindex_v7_to_v8.py", "--allow-missing", "--allow-missing-fraction"),
    (
        "verify_catalog.py",
        "--allow-values-outside-snapshot",
        "--allow-values-outside-snapshot-fraction",
    ),
]

# The other required arguments, so argparse reaches the tolerance check. None of
# these paths is opened before the tolerance is validated, and none of these
# runs reaches a cluster.
OTHER_REQUIRED = {
    "inject_category_path.py": ["--index", "i", "--ndjson", "/nonexistent.ndjson"],
    "reindex_v7_to_v8.py": ["--source", "a", "--dest", "b"],
    "verify_catalog.py": [
        "--ndjson",
        "/nonexistent.ndjson",
        "--taxonomy",
        "/nonexistent.json",
    ],
}


def run_script(script: str, extra: List[str]) -> subprocess.CompletedProcess:
    """Run a script the way the README does: ``python scripts/<name>.py``.

    Which also exercises the one assumption the shared module rests on — that a
    loose script can ``from tolerance import Tolerance`` because Python puts the
    script's own directory on ``sys.path``. Importing the module in-process the
    way the tests above do would not test that.
    """
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script)] + OTHER_REQUIRED[script] + extra,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


@pytest.mark.parametrize("script, count_flag, fraction_flag", CALLER_FLAGS)
def test_the_flags_a_report_names_are_flags_that_caller_defines(
    script: str, count_flag: str, fraction_flag: str
) -> None:
    """``describe()`` quotes strings; nothing stops them naming a phantom flag.

    So the spellings each caller hands the shared class are checked against the
    parser that caller actually builds, by asking it for its own help.
    """
    helped = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert helped.returncode == 0, helped.stderr
    assert count_flag in helped.stdout
    assert fraction_flag in helped.stdout


@pytest.mark.parametrize("script, count_flag, fraction_flag", CALLER_FLAGS)
def test_a_bad_tolerance_is_refused_on_the_command_line_by_its_own_name(
    script: str, count_flag: str, fraction_flag: str
) -> None:
    """Exit 2 and the caller's own flag in the message, not the shared one's.

    ``pytest.raises(SystemExit)`` with a non-zero code would pass on any parse
    error at all — including argparse refusing a flag that no longer exists —
    so the message is asserted, not just the status.
    """
    negative = run_script(script, [count_flag, "-1"])
    assert negative.returncode == 2
    assert f"{count_flag} must not be negative" in negative.stderr

    out_of_range = run_script(script, [fraction_flag, "1.5"])
    assert out_of_range.returncode == 2
    assert f"{fraction_flag} must be between 0 and 1" in out_of_range.stderr
