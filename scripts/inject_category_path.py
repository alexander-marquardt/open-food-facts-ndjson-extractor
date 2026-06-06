#!/usr/bin/env python3
"""Bulk-add the hierarchical ``category_path`` field onto an existing index.

Reads an extractor NDJSON (which contains ``id`` = GTIN and ``category_path``)
and issues partial ``_update`` operations keyed by ``_id`` (= GTIN on PRISM
catalog indexes). Partial updates merge the one field and **do not** run any
ingest pipeline, so existing fields — including the copied embedding vectors —
are left untouched.

Connection comes from the environment:
    PRISM_ELASTICSEARCH_URL, PRISM_ELASTICSEARCH_API_KEY

Usage:
    python scripts/inject_category_path.py \
        --index catalog_en_v8 \
        --ndjson data/products/off_en_hierarchy_full.ndjson
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _es() -> tuple[str, str]:
    url = os.environ.get("PRISM_ELASTICSEARCH_URL", "").rstrip("/")
    key = os.environ.get("PRISM_ELASTICSEARCH_API_KEY", "")
    if not url or not key:
        sys.exit("PRISM_ELASTICSEARCH_URL and PRISM_ELASTICSEARCH_API_KEY must be set")
    return url, key


def _bulk(lines: list[str]) -> dict:
    url, key = _es()
    payload = ("\n".join(lines) + "\n").encode()
    req = urllib.request.Request(f"{url}/_bulk", data=payload, method="POST")
    req.add_header("Authorization", f"ApiKey {key}")
    req.add_header("Content-Type", "application/x-ndjson")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", required=True, help="e.g. catalog_en_v8")
    ap.add_argument("--ndjson", required=True, help="extractor output with category_path")
    ap.add_argument("--batch", type=int, default=1000, help="docs per bulk request")
    ap.add_argument(
        "--skip-empty",
        action="store_true",
        default=True,
        help="do not send updates for records with an empty category_path",
    )
    args = ap.parse_args()

    sent = updated = missing = empty = errors = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal updated, missing, errors
        if not batch:
            return
        try:
            resp = _bulk(batch)
        except urllib.error.HTTPError as exc:
            sys.exit(f"bulk failed: HTTP {exc.code}\n{exc.read().decode(errors='replace')}")
        for item in resp.get("items", []):
            res = item.get("update", {})
            status = res.get("status", 0)
            if status in (200, 201):
                updated += 1
            elif status == 404:
                missing += 1
            else:
                errors += 1
                if errors <= 3:
                    print(f"  error: {json.dumps(res)[:200]}", file=sys.stderr)
        batch.clear()

    with open(args.ndjson, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            gtin = str(doc.get("id") or "").strip()
            path = doc.get("category_path") or []
            if not gtin:
                continue
            if args.skip_empty and not path:
                empty += 1
                continue
            batch.append(json.dumps({"update": {"_id": gtin, "_index": args.index}}))
            batch.append(json.dumps({"doc": {"category_path": path}}))
            sent += 1
            if len(batch) >= args.batch * 2:
                flush()
                if sent % (args.batch * 20) == 0:
                    print(f"  sent {sent:,} (updated {updated:,}, missing {missing:,})", flush=True)
    flush()

    print(
        f"Done. sent={sent:,} updated={updated:,} missing(not in index)={missing:,} "
        f"empty(skipped)={empty:,} errors={errors:,}"
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
