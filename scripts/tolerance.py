"""The one implementation of the scripts' floor-never-round allowance rule.

Three of the tools in this directory gate on a count they measured — ids the
index does not hold, documents a reindex did not copy, catalog values the pinned
snapshot does not explain — and each lets the operator name an exception to that
gate on the command line, as either a whole number or a fraction of whatever the
run actually counted. Each carried its own copy of the rule. The copies were
identical, which is exactly the problem: a change to the rule could land in two
of them and miss the third, and every script would still run and still print a
report, so nothing would say so.

The rule, in one place, is:

* **A whole number wins over a fraction.** They are mutually exclusive on every
  command line that offers them, so at most one is ever set; the order is stated
  anyway rather than left to argparse.
* **A fraction is floored, never rounded.** ``0.5`` of 5 permits 2, not 3.
  Rounding up would let a fraction quietly buy a violation nobody named, which
  is the opposite of what naming a tolerance on a command line is for.
* **Nothing named means zero.** Not "a small number", not "whatever the last run
  had" — the gates in this directory are zero-tolerance by default and an
  exception exists only because somebody typed one.
* **The record says what was permitted, not what was requested.** ``describe()``
  renders the fraction *and* the whole number it floored to against this run's
  denominator, so a report read six months later does not have to recompute it.

What is *not* shared is the naming: each script spells its own two flags, and
:meth:`Tolerance.describe` and :meth:`Tolerance.problem` quote the spelling they
were given so a report and an error message name the flag the operator would
actually type. That is why the flag names are constructor arguments rather than
something this module invents.

Why this lives in ``scripts/`` rather than in ``src/off_demo_extract/``
----------------------------------------------------------------------
``src/off_demo_extract`` is the extractor library: it has no argparse anywhere
in it and no notion of a command line, and this rule is entirely about what an
operator named on one. Putting it there would also cost every importer the
``sys.path`` insert and the lint suppression that ``verify_catalog.py`` and
``build_manifest.py`` need to reach the package from a loose script.

Here, no import machinery is needed at all: Python puts the directory of the
script being run at the front of ``sys.path``, so ``python scripts/whatever.py``
can ``from tolerance import Tolerance`` as an ordinary top-of-file import, and
the tests reach it the same way they already reach the scripts themselves. The
price, stated because it is a real one, is that these tools are no longer a
single file each — copying one somewhere else now means copying this sibling
with it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Tolerance:
    """How much of a gate failure the operator has explicitly agreed to.

    ``count_flag`` and ``fraction_flag`` are the spellings of the two options
    the owning script offers, e.g. ``"--allow-missing"`` and
    ``"--allow-missing-fraction"``. They exist so that a report and a rejected
    command line name a flag that script's parser actually defines.

    ``allow`` and ``allow_fraction`` are the parsed values of those two options,
    ``None`` for "not given". Both being ``None`` is the default and means zero.
    """

    count_flag: str
    fraction_flag: str
    allow: Optional[int] = None
    allow_fraction: Optional[float] = None

    def permitted(self, counted: int) -> int:
        """How many violations are permitted out of ``counted`` things measured.

        ``counted`` is the run's own denominator — ids sent, source documents,
        distinct values checked — and only a fraction tolerance uses it.
        """
        if self.allow is not None:
            return self.allow
        if self.allow_fraction is not None:
            # Floored, never rounded: 0.5 of 5 permits 2, not 3. Rounding up
            # would let a fraction quietly buy a violation nobody named.
            return math.floor(self.allow_fraction * counted)
        return 0

    def describe(self, counted: int) -> str:
        """The tolerance in force, for the report, as what it *permitted*.

        A fraction is rendered with the whole number it floored to against this
        run's denominator, because the number that decided the exit status is
        the one the record has to carry.
        """
        if self.allow is not None:
            return f"{self.count_flag} {self.allow}"
        if self.allow_fraction is not None:
            return (
                f"{self.fraction_flag} {self.allow_fraction:g} "
                f"({self.permitted(counted):,} of {counted:,})"
            )
        return "0 (default: zero tolerance)"

    def problem(self) -> Optional[str]:
        """Why this tolerance is not a tolerance, or ``None`` if it is fine.

        Returned rather than raised so the caller can hand it straight to
        ``ArgumentParser.error``, which is where every one of these comes from:
        the values are read off a command line, so a bad one is a usage error
        and not an exception the script should carry up its own stack.
        """
        if self.allow is not None and self.allow < 0:
            return f"{self.count_flag} must not be negative"
        if self.allow_fraction is not None and not 0.0 <= self.allow_fraction <= 1.0:
            return f"{self.fraction_flag} must be between 0 and 1"
        return None
