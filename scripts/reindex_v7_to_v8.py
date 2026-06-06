#!/usr/bin/env python3
"""Copy an existing PRISM catalog index to a new ``_v8`` *without re-embedding*.

The embedding vectors (``content_embedding_elser`` / ``_e5`` / ``_jina``) are
physically stored in ``_source`` on the v7 indexes, so a server-side ``_reindex``
with **no ingest pipeline** copies them verbatim — no inference is run. The new
index is created from the source's own mapping/settings, minus the per-index
``default_pipeline`` (which is what would otherwise re-embed), plus a new
``category_path`` keyword field for the hierarchical category facet.

Connection comes from the environment:
    PRISM_ELASTICSEARCH_URL, PRISM_ELASTICSEARCH_API_KEY

Usage:
    python scripts/reindex_v7_to_v8.py --source catalog_en_v7 --dest catalog_en_v8
    python scripts/reindex_v7_to_v8.py --source catalog_en_v7 --dest catalog_en_v8 --recreate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request


# Settings keys that are index-instance-specific and must not be carried to a new index.
_DROP_SETTINGS = {
    "uuid",
    "creation_date",
    "version",
    "provided_name",
    "routing",
    "default_pipeline",  # <- the whole point: dropping this prevents re-embedding
    "history",
}


def _es() -> tuple[str, str]:
    url = os.environ.get("PRISM_ELASTICSEARCH_URL", "").rstrip("/")
    key = os.environ.get("PRISM_ELASTICSEARCH_API_KEY", "")
    if not url or not key:
        sys.exit("PRISM_ELASTICSEARCH_URL and PRISM_ELASTICSEARCH_API_KEY must be set")
    return url, key


def _req(method: str, path: str, body: dict | None = None) -> dict:
    url, key = _es()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url + path, data=data, method=method)
    req.add_header("Authorization", f"ApiKey {key}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"{method} {path} -> HTTP {exc.code}\n{detail}") from exc


def _exists(index: str) -> bool:
    url, key = _es()
    req = urllib.request.Request(f"{url}/{index}", method="HEAD")
    req.add_header("Authorization", f"ApiKey {key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def build_dest_body(source: str) -> dict:
    """Construct the create-index body from the source's mapping + settings."""
    mapping = _req("GET", f"/{source}/_mapping")[source]["mappings"]
    settings_idx = _req("GET", f"/{source}/_settings")[source]["settings"]["index"]

    clean_settings = {
        k: v for k, v in settings_idx.items() if k not in _DROP_SETTINGS
    }

    # Add the hierarchical category field. Keyword for exact faceting / drill-down;
    # a `.text` sub-field mirrors how customer catalogs (e.g. Musgrave) expose the
    # category for optional full-text boosting later.
    props = dict(mapping.get("properties", {}))
    if "category_path" not in props:
        props["category_path"] = {
            "type": "keyword",
            "fields": {"text": {"type": "text", "analyzer": "base"}},
        }
    new_mapping = dict(mapping)
    new_mapping["properties"] = props

    return {"settings": {"index": clean_settings}, "mappings": new_mapping}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="e.g. catalog_en_v7")
    ap.add_argument("--dest", required=True, help="e.g. catalog_en_v8")
    ap.add_argument("--recreate", action="store_true", help="delete dest if it exists")
    ap.add_argument("--no-wait", action="store_true", help="don't poll reindex task")
    args = ap.parse_args()

    if _exists(args.dest):
        if not args.recreate:
            sys.exit(f"{args.dest} already exists (use --recreate to replace)")
        print(f"Deleting existing {args.dest} ...")
        _req("DELETE", f"/{args.dest}")

    src_count = _req("GET", f"/{args.source}/_count")["count"]
    print(f"Source {args.source}: {src_count:,} docs")

    body = build_dest_body(args.source)
    has_dp = "default_pipeline" in body["settings"]["index"]
    print(
        f"Creating {args.dest} (default_pipeline carried over: {has_dp}; "
        f"category_path added: {'category_path' in body['mappings']['properties']})"
    )
    _req("PUT", f"/{args.dest}", body)

    print("Starting server-side reindex (no pipeline -> embeddings copied as-is) ...")
    task = _req(
        "POST",
        "/_reindex?wait_for_completion=false&slices=auto&refresh=false",
        {"source": {"index": args.source}, "dest": {"index": args.dest, "op_type": "create"}},
    )
    task_id = task.get("task")
    print(f"Reindex task: {task_id}")
    if args.no_wait:
        print(f"Poll with: GET /_tasks/{task_id}")
        return 0

    while True:
        time.sleep(5)
        status = _req("GET", f"/_tasks/{task_id}")
        st = status.get("task", {}).get("status", {})
        created = st.get("created", 0)
        total = st.get("total", src_count)
        print(f"  reindexed {created:,}/{total:,}", flush=True)
        if status.get("completed"):
            failures = status.get("response", {}).get("failures", [])
            if failures:
                print(f"FAILURES: {json.dumps(failures[:3], indent=2)}")
                return 1
            break

    _req("POST", f"/{args.dest}/_refresh")
    dst_count = _req("GET", f"/{args.dest}/_count")["count"]
    print(f"Done. {args.dest}: {dst_count:,} docs (source had {src_count:,})")
    return 0 if dst_count == src_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
