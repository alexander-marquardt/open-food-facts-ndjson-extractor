# Index verification — `catalog_{en,es,fr}_v13` against the 2026-08-03 manifest

[`VERIFICATION.md`](VERIFICATION.md) verified the *files* this build produced and
ended by naming what it could not do: the run was required to stay offline, so
nothing compared the numbers it pinned against what is actually in an index, and
"the fr load did not complete" could not be told apart from "the index
population predates the fr extract".

This is that measurement. It was taken with `scripts/verify_index.py`, read-only,
against the live cluster. The machine-readable form of every number below is
`index_verify_{lang}_v13.json`, committed next to this file verbatim.

```bash
python scripts/verify_index.py \
    --index catalog_fr_v13 \
    --manifest builds/2026-08-03/build_manifest.json \
    --taxonomy <taxonomy>/categories.json \
    --catalog <catalogs>/off_fr_v14.ndjson \
    --json builds/2026-08-03/index_verify_fr_v13.json
```

## Document count

| | manifest `distinct_ids` | index `_count` | delta |
| :--- | ---: | ---: | ---: |
| `catalog_en_v13` | 108,379 | 107,451 | **−928** |
| `catalog_es_v13` | 31,910 | 31,910 | **0** |
| `catalog_fr_v13` | 222,955 | 195,209 | **−27,746** |

es reconciles to the digit. The other two do not, and — this is the part a count
alone cannot give you — they do not disagree for the same reason.

## The fr shortfall: the load did not complete

Three measurements, none of which needed a scroll:

1. **Every indexed fr document carries `category_path`** — 195,209 of 195,209.
   So no document was written and then missed by the partial-update pass. Had
   `inject_category_path.py` run over an index holding documents the extract does
   not have, those documents would still be there, without the field.
2. **Zero index ids are absent from the extract.** The index is a strict subset
   of `off_fr_v14.ndjson`, not a differently-populated set that happens to be
   smaller. (en and es are strict subsets too — 0 and 0.)
3. **The missing ids arrive in blocks.** 27,746 catalog ids are absent from the
   index, in **48 contiguous runs** in catalog order. Thirteen of those runs hold
   **27,707** of them — 99.86% — the largest being 5,936 consecutive records:

   | catalog position | length |
   | ---: | ---: |
   | 61,861 | 5,936 |
   | 33,116 | 4,954 |
   | 218,913 | 4,042 |
   | 38,748 | 2,776 |
   | 68,292 | 2,461 |
   | 11,795 | 1,977 |
   | 160,317 | 1,955 |

Thousands of *consecutive* extract records missing together is the signature of
bulk batches that failed and were never retried. It is not what an index built
from a different extract looks like: a difference between two extract
generations is decided per record — by whether that product's tags resolve — and
lands scattered.

The contrast with en, in the same measurement, is what makes that argument
concrete rather than plausible:

| | missing ids | contiguous runs | runs ≥ 100 | ids in those runs | largest run |
| :--- | ---: | ---: | ---: | ---: | ---: |
| en | 928 | 700 | 0 | 0 | 13 |
| fr | 27,746 | 48 | 13 | 27,707 | 5,936 |

**en's 928 is re-extraction drift, not a load failure.** 928 ids over 700 runs,
none longer than 13, is exactly the per-record shape — and it is the same
population `VERIFICATION.md` already accounts for: the v14 pipeline resolves a
path for 926 en products that previously had none, so the v14 extract emits
products the v13-generation load never had to write. es changed by 0 there and is
short by 0 here.

So of issue #12's two candidate mechanisms, the answer for fr is the **first**:
the fr load did not complete. Nothing about the extract, the taxonomy or
`inject_category_path.py`'s update semantics is implicated.

**What this does not establish.** These are v13 indices measured against a v14
manifest, because no v14 index exists yet — the indices were loaded from an
earlier extract generation. The id-set comparison is therefore against the
nearest available extract, not the exact file that was loaded. That weakens
nothing about the block structure (a per-record difference between two extract
generations cannot produce 5,936 consecutive absences), but it does mean the
27,746 figure is "short against the current extract", not "short against the file
someone ran".

## Category vocabulary

Every distinct `categories` value and every `category_path` segment in each
index, checked against the display labels of the pinned snapshot — the same rule
`verify_catalog.py` applies to the NDJSON, applied to the other end.

| | distinct `categories` in index | outside the snapshot | instances | snapshot labels unused | path heads that are not global roots |
| :--- | ---: | ---: | ---: | ---: | ---: |
| en | 11,068 | **6,904** | 120,142 | 10,289 | 61 of 114 |
| es | 5,046 | **4,032** | 197,278 | 13,439 | 10 of 55 |
| fr | 16,743 | **15,795** | 1,178,189 | 13,504 | 27 of 83 |

All three fail, and they should: **the v13 indices predate every pipeline change
this build made**. Two kinds of divergence show up plainly in the top offenders:

* **Label rendering.** `Plant based foods and beverages` (38,323 en documents) is
  not in the snapshot; the snapshot renders that node `Plant-based foods and
  beverages`. A set difference is the only thing that notices a hyphen.
* **Language.** The es index files products under `Plant based foods and
  beverages`, `Snacks`, `Sweet snacks` — English labels in a Spanish catalog.
  The v14 extract emits `Bebidas`, `Bebidas carbonatadas`, `Sodas`. That is
  language-blind traversal with the language filter at the leaf (#22) landing.

`Groceries` (6,299 en documents) is the value issue #12 named, and it is still
there: it is not a taxonomy node in any snapshot, upstream or pinned. The v14
catalogs carry **0** values outside the snapshot, so a rebuilt index would green
this row.

Structural checks on `category_path` pass everywhere: **0** segments outside the
snapshot and **0** orphan addresses (an address whose one-segment-shorter prefix
never reached the index) in all three. What fails is anchoring — 61 of en's 114
distinct path heads are not display labels of one of the 92 global roots — which
is the pre-#22 traversal choosing a different root, exactly the churn
`VERIFICATION.md` measures at 32–38% of products.

## Manifest identity: unverifiable, and why

Every one of the three indices reports `unverifiable`, not pass or fail. Their
mappings carry no `_meta`, so an index cannot be asked which build produced it;
the manifest named on the command line is an assertion by the operator. Recording
that as a pass would be a lie, and recording it as a failure would train people
to ignore it.

The remedy is small and needs no reindex — `_meta` is free-form, survives
`dynamic: strict`, and can be added to a live index with
`PUT /<index>/_mapping`:

```json
"_meta": {"off_catalog_build": {
    "manifest_schema": "off-catalog-build-manifest/1",
    "manifest_sha256": "...",
    "generated_utc": "2026-08-03T10:34:52Z",
    "lang": "fr",
    "extractor_commit": "9e8a4c6f38e7f03d7f42dc4fbc97601210285d83",
    "dump_sha256": "f06f34f7...",
    "taxonomy_sha256": "74717ecc...",
    "catalog_sha256": "a97e1826...",
    "expected_distinct_ids": 222955
}}
```

Once the loader writes that, `verify_index.py` stops needing to be *told* which
manifest to trust: it reads the claim off the index and refuses a manifest that
disagrees. Until then, the one identity fact it can still establish is that the
`--taxonomy` file handed to it is byte-for-byte the snapshot the manifest pins —
which it verified here (`74717ecc…`), so the vocabulary rows above are known to
be judged against the right snapshot.

## Cost

The default run — count, coverage, identity, and both vocabularies — is **two
requests**: one `_mapping` read and one `size: 0` search carrying
`track_total_hits`, the coverage filter and both terms aggregations. Adding
`--catalog` costs one pass over the NDJSON plus a paginated `composite`
enumeration of the index's ids (~20 requests for fr, ~42 s). Nothing scrolls, and
no request can reach a write endpoint.

The terms aggregations were checked for truncation rather than assumed complete:
`buckets_returned` (4,601 / 3,173 / 6,499 for `category_path`) never reached the
requested 30,000, and `sum_other_doc_count` and `doc_count_error_upper_bound`
were both 0. Forcing truncation with `--terms-size 100` on es makes the run
escalate to the exhaustive `composite` enumeration and produce **identical**
vocabulary numbers, which is the positive control that the escalation is real
rather than decorative.

## Recommendation, unchanged

Rebuild the indices from the v14 NDJSON rather than injecting fields into these,
and run `verify_index.py` against the result before switching anything over.
Every row above is expected to green on a rebuilt index except `manifest_identity`,
which greens only once the loader writes `_meta`.

The fr reload is gated on a cluster-scaling decision and is out of scope here.
