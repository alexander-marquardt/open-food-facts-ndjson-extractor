#!/usr/bin/env python3
"""Record the identity of a catalog build so it can be reproduced or refuted.

A catalog is a function of three things: the product dump, the category taxonomy
snapshot, and the extractor commit. None of them is self-describing once the
NDJSON is indexed, so a mismatch between what is in a cluster and what this
repository would build today is only discoverable by hand. This script writes a
manifest that pins all three by checksum, alongside the counts each locale run
produced, so drift is a diff rather than an archaeology exercise.

It reads the extractor's own per-locale report and (optionally) the artifact-side
output of ``scripts/verify_catalog.py``; it computes checksums itself rather than
accepting them as arguments, so the manifest cannot claim a digest nobody ran.
It never contacts a cluster.

Usage:
    python scripts/build_manifest.py \\
        --dump data/json_source/openfoodfacts-products.jsonl.gz \\
        --taxonomy data/json_source/categories.json \\
        --locale en:report_en.json:verify_en.json:off_en.ndjson \\
        --locale fr:report_fr.json:verify_fr.json:off_fr.ndjson \\
        --out builds/2026-08-03/build_manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from off_demo_extract.taxonomy import global_roots, load_taxonomy  # noqa: E402

CHUNK = 1 << 22


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_identity(path: Path, *, digest: bool = True) -> Dict[str, Any]:
    stat = path.stat()
    identity: Dict[str, Any] = {
        "name": path.name,
        "bytes": stat.st_size,
        "modified_utc": _utc(stat.st_mtime),
    }
    if digest:
        identity["sha256"] = sha256(path)
    return identity


def git_identity(repo: Path) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    # Untracked files are excluded on purpose: the manifest is itself written to
    # an untracked path, so counting them would make "clean" unreachable. What
    # matters for reproducibility is whether the *tracked* source that ran
    # differs from the commit being recorded.
    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": run("rev-parse", "HEAD"),
        "described": run("describe", "--always"),
        "tracked_files_clean": status == "" if status is not None else None,
    }


def _load(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def locale_entry(spec: str) -> Dict[str, Any]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"--locale needs at least lang:report, got {spec!r}")
    lang = parts[0]
    report = _load(Path(parts[1])) or {}
    verify = _load(Path(parts[2])) if len(parts) > 2 and parts[2] else None
    catalog = Path(parts[3]) if len(parts) > 3 and parts[3] else None

    counters = report.get("counters", {})
    curation = report.get("category_tag_curation", {})
    entry: Dict[str, Any] = {
        "lang": lang,
        "catalog": file_identity(catalog, digest=True) if catalog else None,
        "elapsed_seconds": report.get("elapsed_seconds"),
        "counters": counters,
        # The three properties the rebuild exists to establish, kept at the top
        # level of the entry so a reader does not have to know which nested block
        # each one lives in.
        "category_path_anchoring": report.get("category_path_anchoring"),
        "category_path_addresses": report.get("category_path_addresses"),
        "refusals": {
            key: curation.get(key)
            for key in (
                "tag_instances",
                "accepted_instances",
                "aliased_instances",
                "rejected_instances",
                "rejected_rate",
                "rejected_by_reason",
                "unknown_tag_instances",
                "unknown_tag_rate",
                "distinct_unknown_tags",
                "products_with_rejected_tags",
                "products_with_no_accepted_tag",
                "top_unknown_tags",
            )
        },
        "artifact_verification": verify,
    }
    return entry


def main(argv: Optional[List[str]] = None) -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument(
        "--dump-source",
        default="https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz",
        help="where the dump came from, recorded verbatim",
    )
    parser.add_argument(
        "--locale",
        action="append",
        default=[],
        metavar="LANG:REPORT[:VERIFY[:CATALOG]]",
        help="repeatable; one per locale built in this run",
    )
    parser.add_argument("--note", default=None, help="free-text note recorded with the build")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    taxonomy = load_taxonomy(args.taxonomy)
    manifest = {
        "schema": "off-catalog-build-manifest/1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "extractor": git_identity(repo),
        "dump": {**file_identity(args.dump), "source": args.dump_source},
        "taxonomy": {
            **file_identity(args.taxonomy),
            "nodes": len(taxonomy),
            "global_roots": len(global_roots(taxonomy)),
        },
        "locales": [locale_entry(spec) for spec in args.locale],
    }
    if args.note:
        manifest["note"] = args.note

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
