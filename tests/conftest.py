"""Make ``src/off_demo_extract`` importable from the test modules.

This package is a source checkout, not an installed distribution: there is no
editable install step in the README's workflow, so ``off_demo_extract`` is not
on ``sys.path`` when pytest collects ``tests/``.

Putting the path setup HERE rather than at the top of each test module is what
lets every test import the package the ordinary way, on the first lines of the
file. pytest imports ``conftest.py`` before it imports any test module in the
same directory, so the path is already in place by the time an import runs.
The alternative — an ``sys.path.insert`` above the imports inside each test
file — makes those imports non-top-of-file, which needs a lint suppression on
every one of them; this project does not use suppressions, so the import
arrangement has to be correct instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
