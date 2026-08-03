# Catalog rebuild — 2026-08-03, `taxonomy_tags`

Re-extraction of the `en`, `es` and `fr` catalogs after renaming the flat
category tag field from `categories` to `taxonomy_tags`, against the **same
pinned inputs** as the [2026-08-03 build](../2026-08-03/VERIFICATION.md).

The point of this build is not new data. It is to prove that **the only
difference is the field name** — that renaming a key moved nothing else. It is
**offline**: nothing here contacted, read or wrote a cluster, and no index was
built or changed.

The machine-readable form is [`build_manifest.json`](build_manifest.json).

## Inputs, pinned — identical to the previous build

| | value | same as before? |
| :--- | :--- | :--- |
| Dump SHA-256 | `f06f34f7ecd19405bf3e91a31d638d96ba91cd364bee69f9530a6c6380dd2f5f` | yes |
| Taxonomy SHA-256 | `74717ecc001cf8661f6ec0bb3fc8c7a0cf317a6355a245004e892348fe575ec5` | yes |
| Taxonomy nodes / global roots | 14,457 / 92 | yes |
| Records read per locale | 4,241,020 | yes |
| Extractor commit | `b90ba8e` (the rename) | no — this is the change under test |
| Wall clock, all three in parallel | **9 min 12 s** (2026-08-03T14:46:23Z → 14:55:35Z) | previous: 8 min 26 s |
| Per-locale elapsed | en 548.7 s, es 536.1 s, fr 551.5 s | previous: 498.9 / 498.9 / 506.0 |

The ~46 s of extra wall clock is machine contention, not the pipeline: the
extraction is single-threaded per locale and the record-level output is proven
identical below. It is reported as measured rather than explained away.

## The delta is the field name, and nothing else

Every figure from the previous build reproduces **to the digit**:

| | en | es | fr |
| :--- | ---: | ---: | ---: |
| Records written | 108,380 | 31,913 | 223,036 |
| Distinct ids | 108,379 | 31,910 | 222,955 |
| Categories at multiple addresses | 0 | 0 | 0 |
| Property 3 violations | 0 | 0 | 0 |
| Traversal roots | 92 | 92 | 92 |
| Unanchored / phantom roots | 0 / 0 | 0 / 0 | 0 / 0 |
| Rejected tags | 23,982 | 7,609 | 49,712 |

This is not just the headline table. **Every** integer counter in the previous
build's manifest was diffed against this one — `counters`,
`category_path_anchoring`, `category_path_addresses`, `refusals` (including the
fractional `rejected_rate` and `unknown_tag_rate`) and `artifact_verification`.
Zero discrepancies.

### Record-level identity — the check that actually settles it

Counts agreeing is weak evidence: two different catalogs can have the same
counts. The strong check takes the new NDJSON, renames `taxonomy_tags` back to
`categories` in every record, and compares against the old file.

| | en | es | fr |
| :--- | ---: | ---: | ---: |
| Records compared | 108,380 | 31,913 | 223,036 |
| **Byte-identical after re-keying** | **108,380 / 108,380** | **31,913 / 31,913** | **223,036 / 223,036** |
| Key-order differences | 0 | 0 | 0 |

The re-keyed stream reproduces the **previous build's SHA-256 exactly**:

| | previous catalog SHA-256 | new catalog, key renamed back |
| :--- | :--- | :--- |
| en | `06cb9b1a1cd4af706aa9c35b9e5cf7e6a87fb9bfa6913bb5679ed9f6b44e7c87` | identical |
| es | `c16e72d6b459a603875d5962a5179db0d9856e41d7f455a2fab9308de15940ee` | identical |
| fr | `a97e1826e0d66f2cdf875b057cdae329289c5e304808118e521bffb5aa47574a` | identical |

Not "no differences found" — the same digest. Every byte of every record, in
order, including key order within each JSON object.

### The file sizes close arithmetically too

`taxonomy_tags` is 3 characters longer than `categories`, once per record:

| | before → after | delta | records × 3 |
| :--- | ---: | ---: | ---: |
| en | 271,865,424 → 272,190,564 | 325,140 | 108,380 × 3 = 325,140 |
| es | 81,518,709 → 81,614,448 | 95,739 | 31,913 × 3 = 95,739 |
| fr | 584,894,152 → 585,563,260 | 669,108 | 223,036 × 3 = 669,108 |

Every added byte is accounted for by the name itself. There is no room in the
file for a second change.

## `scripts/verify_catalog.py`

Run against all three new catalogs, reading the field under its new name.
**Exit status 0** for each; `=> all checks passed`.

| | en | es | fr |
| :--- | ---: | ---: | ---: |
| Property 3 violations | 0 | 0 | 0 |
| Property 2 — categories at 2+ addresses | 0 | 0 | 0 |
| Chains not headed by a taxonomy root | 0 | 0 | 0 |
| Distinct values checked against the snapshot | 4,605 | 3,263 | 6,502 |
| Values outside the pinned snapshot | 0 | 0 | 0 |
| Duplicate id instances (reported, not gated) | 1 | 3 | 81 |

The vocabulary check is the one that would have caught a half-done rename: it
reads the flat field out of the NDJSON by name. Had `verify_catalog.py` been
left on the old key it would have found no values there and checked only path
segments — the value counts above (4,605 / 3,263 / 6,502, matching the previous
build's `labels_seen`) are what shows it is reading the field it thinks it is.

## Where the catalogs are

Not in this repository — `.gitignore` excludes `data/*` and `*.ndjson`. The
previous catalogs are **retained** alongside, deliberately, for comparison.

```
~/Documents/off-catalogs-2026-08-03-tags/
    off_en_v14.ndjson      272,190,564 B   sha256 03e00298efdf8319fb8a29de58b33b5bec7b8d3be6d1b8b40dfe827b1aff085d
    off_es_v14.ndjson       81,614,448 B   sha256 67efe89523e49d7fab195fe3dec003464f7a494403d2377044f58567b5797734
    off_fr_v14.ndjson      585,563,260 B   sha256 96996b447540de964a067485185731bfe5f9837aca3821f0f81980070bf24662
    report_{lang}_v14.json  the extractor's own run report
    verify_{lang}_v14.json  the artifact-side verification
    stderr_{lang}.log       the full run log, including progress lines
    build_manifest.json     a copy of the manifest committed here
```

## What this build does not do

* **No index was written**, and no cluster was contacted.
* **The dump is still the January one** — deliberately, for the same reason as
  the previous build: holding every input fixed is what makes this a rename
  test rather than a re-derivation.
* **The index-side readers still query `categories`.** `scripts/verify_index.py`,
  `tests/test_index_verification.py` and the envelope fixture read the field
  from Elasticsearch, not from these files. They are correct against every index
  that exists today and must move when an index is first built from these
  catalogs. Tracked in #42.
