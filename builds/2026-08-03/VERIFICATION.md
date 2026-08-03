# Catalog rebuild — 2026-08-03

Re-extraction of the `en`, `es` and `fr` catalogs with the current pipeline,
against a **pinned** taxonomy snapshot, with the identity of every input
recorded. This is the build described by issue #12. It is **offline**: nothing
here contacted, read or wrote a cluster, and no index was built or changed.

The machine-readable form of everything below is
[`build_manifest.json`](build_manifest.json). The extractor's own per-locale
reports (`report_*_v14.json`) and the artifact-side verification
(`verify_*_v14.json`) are committed next to it, verbatim.

## Inputs, pinned

| | value |
| :--- | :--- |
| Dump | `openfoodfacts-products.jsonl.gz` |
| Dump size | 10,489,757,443 bytes |
| Dump modified | 2026-01-09T20:46:03Z |
| Dump SHA-256 | `f06f34f7ecd19405bf3e91a31d638d96ba91cd364bee69f9530a6c6380dd2f5f` |
| Taxonomy | `categories.json` |
| Taxonomy size | 4,532,475 bytes |
| Taxonomy SHA-256 | `74717ecc001cf8661f6ec0bb3fc8c7a0cf317a6355a245004e892348fe575ec5` |
| Taxonomy nodes | 14,457 |
| Taxonomy global roots | 92 |
| Extractor commit | `be659d6` (source unchanged by this build's commits — `git diff be659d6 HEAD -- src/` is empty) |
| Records read per locale | 4,241,020 |
| Wall clock, all three locales in parallel | **8 min 26 s** (2026-08-03T10:23:38Z → 10:32:04Z) |
| Per-locale elapsed | en 498.9 s, es 498.9 s, fr 506.0 s |

The manifest records commit `9e8a4c6`, which adds the two scripts used to verify
and record this build and touches nothing under `src/`. The extraction itself ran
at `be659d6`.

### The dump is the January one, deliberately

The issue asks for a *current* dump. This build used the **local 2026-01-09**
dump and did not download a new one. Measured today by an HTTP `HEAD` (no body
fetched):

| | date | size |
| :--- | :--- | :--- |
| local, used here | 2026-01-09 | 10,489,757,443 B (10.5 GB) |
| upstream | 2026-08-03T06:24:11Z | 12,601,572,416 B (12.6 GB) |

The trade: a fresh dump costs a 12.6 GB download and adds ~7 months of new
products, but it changes *two* variables at once. Every number below would then
be the sum of "the pipeline changed" and "the data changed", and the pipeline
changes are what this rebuild exists to validate. Holding the dump fixed makes
the comparison against the previous catalogs exact — and it is, to the record
(see [What changed](#what-changed-against-the-previous-catalogs)): every count
closes arithmetically. A dump refresh is a clean second pass against this
manifest, and the manifest is what makes it a diff rather than a re-derivation.

## Verification, per locale

Two independent readings for each property: the extractor's own report, and
`scripts/verify_catalog.py` re-deriving it from the written NDJSON. They agree on
every row.

| | en | es | fr |
| :--- | ---: | ---: | ---: |
| Records written | **108,380** | **31,913** | **223,036** |
| Distinct ids (what an index count must match) | 108,379 | 31,910 | 222,955 |
| Duplicate id instances | 1 | 3 | 81 |
| `with_category_path` | 108,380 | 31,913 | 223,036 |
| Records with an empty path | 0 | 0 | 0 |
| `taxonomy_roots` | **92** | **92** | **92** |
| `traversal_roots` | **92** | **92** | **92** |
| `phantom_roots` | 0 | 0 | 0 |
| `unanchored_category_path` | **0** | **0** | **0** |
| Unanchored chains, re-derived from the file | 0 | 0 | 0 |
| **Categories at multiple addresses** | **0** | **0** | **0** |
| Categories under multiple labels | 0 | 0 | 0 |
| Labels shared by multiple categories | 0 | 0 | 0 |
| Single-chain violations (property 3) | **0** | **0** | **0** |
| Distinct categories used | 3,977 | 2,721 | 5,582 |
| Emitted values absent from the pinned snapshot | 0 | 0 | 0 |
| Mean path depth | 3.52 | 4.05 | 4.14 |

`traversal_roots == taxonomy_roots == 92` with no phantom roots is the forest
invariant holding: the graph the chains walk has exactly the taxonomy's roots,
so no chain can be anchored to a root the taxonomy does not have.

### Property 2 — every category at exactly one address

**0** in all three locales, read two ways: the run's own `AddressAudit`, and a
second pass over the finished NDJSON that maps each path segment to the set of
cumulative addresses it occurs under. A category occupying two addresses would
mean `category_path` is not a tree; none does.

### Property 3 — exactly one chain per product

**0 violations** in all three locales, over 363,329 records. The check is not
"the path is non-empty": `scripts/verify_catalog.py` requires element *i* of
`category_path` to be element *i-1* plus exactly one segment, so a union of
parallel branches, a skipped level or a repeat all fail. Every record's path is a
single root→leaf chain, and its first segment is the display label of one of the
92 global roots.

### Refusal breakdown

| | en | es | fr |
| :--- | ---: | ---: | ---: |
| Tag instances seen | 576,185 | 212,522 | 1,504,955 |
| Accepted | 552,203 | 204,913 | 1,455,243 |
| Aliased through the rename map | 2,362 | 597 | 6,646 |
| **Rejected** | 23,982 (4.16%) | 7,609 (3.58%) | 49,712 (3.30%) |
| — not in the taxonomy | 16,060 | 5,207 | 34,317 |
| — curated drop list | 6,685 | 1,740 | 14,330 |
| — out of language | 1,237 | 662 | 1,065 |
| Distinct unknown tags | 9,805 | 2,591 | 14,854 |
| Products left with no accepted tag | 5,973 | 1,314 | 4,614 |
| Products dropped for no category | 43,710 | 3,850 | 34,148 |

Values are dropped, records are not — except where refusing every one of a
product's tags leaves it with no category at all, which the pre-existing
`--require-category` filter then drops. That is the whole of the record-count
change, and it closes exactly (below).

## What changed against the previous catalogs

Same dump, same taxonomy file, different pipeline. Compared against the June
catalogs (`off_{lang}_hierarchy_full.ndjson`):

| | en | es | fr |
| :--- | ---: | ---: | ---: |
| Records, before → after | 114,353 → 108,380 | 33,227 → 31,913 | 227,650 → 223,036 |
| Products left with no accepted tag | −5,973 | −1,314 | −4,614 |
| Previously path-less, now carrying a path | +926 | +0 | +3 |
| Leaf **category** changed | 28,179 (26.2%) | 9,641 (30.2%) | 64,979 (29.1%) |
| Root changed | 34,218 (31.8%) | 12,164 (38.1%) | 80,119 (35.9%) |
| Full address changed | 66,395 (61.8%) | 23,855 (74.8%) | 157,967 (70.9%) |
| Same leaf, different ancestry | 38,216 (35.6%) | 14,214 (44.5%) | 92,988 (41.7%) |
| Mean path depth | 3.80 → 3.52 | 4.47 → 4.05 | 4.53 → 4.14 |

Every record-count change closes to the digit:

* en: 114,353 − 5,973 = **108,380**. The 6,899 records that previously carried no
  path are exactly the 5,973 dropped for having no accepted tag plus the 926 that
  now resolve.
* es: 33,227 − 1,314 = **31,913**; 1,314 previously path-less, 1,314 dropped, 0
  recovered.
* fr: 227,650 − 4,614 = **223,036**; 4,617 previously path-less = 4,614 dropped +
  3 recovered.

No record was dropped by the anchoring gate in any locale
(`missing_category_path` and `unanchored_category_path` are both 0). The gate
refusing nothing is the expected result and is what shows the forest still has
exactly the taxonomy's roots.

**The address churn is much larger than a re-extraction against a newer taxonomy
would produce, and that is worth stating plainly**: 62–75% of products change
their full address and 26–30% change the category they are filed under. This is
the canonical-parent map doing its job — one parent per category, fewest hops to
a root with a lexicographic tie-break, walked language-blind — replacing a walk
that made a different choice per language and per product. Mean depth *falls*
rather than rises: materialising untagged ancestors lengthens chains, but
fewest-hops shortens them more. Anyone expecting a few percent of movement is
comparing against a single increment of the pipeline work rather than against the
June catalogs, which predate all of it. Nothing in the invariants regressed —
roots, anchoring, addressing and single-chain are all at their required values —
but the facet tree a user sees will be materially different, and that is a
product-visible change to schedule around, not a silent one.

`popularity` and `margin` change shape entirely, as expected from deriving them
from real data instead of a seeded uniform draw:

| | before | after (en) |
| :--- | :--- | :--- |
| `popularity` | uniform 0–10,000, 10,000 distinct values, 0.0% zeros | `1000·ln(1+unique_scans)`, 208 distinct values, **38.8% zeros**, median 693 (= one scanner), max 6,236 |
| `margin` | uniform 0–200, 201 distinct values | per-category modelled rate, 20–59, 30 distinct values, **no zeros**, median 30 |

es and fr show the same shift (fr: 21.9% zero popularity, margin 20–59; es:
22.1% / 20–59).

## The fr discrepancy

Issue #12 records that the fr extract report said `with_category_path: 223,033`
while the fr index holds **195,209**, and that en and es reconcile exactly.

What this run produces: fr **223,036** records written, **222,955** distinct ids.
Essentially unchanged from 223,033 / 222,952 — so the gap is not something the
new pipeline creates or removes.

What the numbers say about where the gap is:

* **The extract side is consistent and always was.** The June fr NDJSON holds
  227,650 records, 223,033 of which carry a path, across 222,952 distinct ids.
  The 223,033 is a field-population count over a file that also contains 4,617
  path-less records; the indices were loaded with path-carrying records only,
  which is exactly what `scripts/inject_category_path.py --skip-empty` sends.
  Under that rule en (107,454) and es (31,913) match their `with_category_path`
  to the digit. fr's 195,209 is **27,824 short** of the same rule — a 12.5%
  shortfall, in the one locale where the file is largest.
* **Duplicate ids do not explain it.** They account for at most 81 documents
  (223,033 records → 222,952 distinct ids), not 27,824.
* **No other file on disk holds 195,209 fr records.** The three older fr
  catalogs all hold 227,650 records, and the two pre-hierarchy ones carry no
  `category_path` at all, so none of them can be the source of a path-carrying
  195,209.

So the shortfall is on the **indexing** side, not in the extract: either the fr
load did not complete, or it was made against an index population that predates
the fr extract. `inject_category_path.py` makes the second easy to do by
accident — it issues partial `_update` operations keyed by id, which can only
touch documents already present, and counts the rest as `missing` rather than
adding them. A locale whose index population predates its extract therefore ends
up with fewer documents than the extract has, silently.

**Not verified here**: this run was required to stay offline, so the index count
and the id-level diff were not read from a cluster. Distinguishing "the load
stopped early" from "the index predates the extract" needs one `_count` and one
id sample, and that is the natural first step of the indexing pass.

> **Since measured.** `scripts/verify_index.py` read the cluster and answered it:
> the fr load did not complete. The 27,746 ids the index lacks arrive in 48
> contiguous runs, 13 of which hold 99.86% of them and the largest of which is
> 5,936 consecutive records — the shape of dropped bulk batches, not of an index
> built from a different extract. See
> [`INDEX_VERIFICATION.md`](INDEX_VERIFICATION.md).

Two things make the class of error non-recurring rather than merely explained:

1. `--require-category-path` is on by default, so `written == with_category_path`
   in all three locales. There is no longer a second, smaller number in the
   report that an index could be compared against by mistake.
2. The manifest pins the number an index must hold — **distinct ids**, not
   records: en 108,379, es 31,910, fr 222,955. A post-index `_count` that does
   not match those is now a one-line check rather than an archaeology exercise.

Recommendation for the indexing pass: **rebuild** the indices from these NDJSON
files rather than injecting fields into the existing ones, and compare the
resulting counts against the manifest before switching anything over to them.

## Where the catalogs are

They are **not** in this repository — `.gitignore` excludes `data/*` and
`*.ndjson`, and the only NDJSON the repository tracks is the 18 KB sample under
`data/sample-data/`. These are 100–600 MB each and stay out.

```
~/Documents/off-catalogs-2026-08-03/
    off_en_v14.ndjson      271,865,424 B   sha256 06cb9b1a1cd4af706aa9c35b9e5cf7e6a87fb9bfa6913bb5679ed9f6b44e7c87
    off_es_v14.ndjson       81,518,709 B   sha256 c16e72d6b459a603875d5962a5179db0d9856e41d7f455a2fab9308de15940ee
    off_fr_v14.ndjson      584,894,152 B   sha256 a97e1826e0d66f2cdf875b057cdae329289c5e304808118e521bffb5aa47574a
    report_{lang}_v14.json  the extractor's own run report
    verify_{lang}_v14.json  the artifact-side verification
    stderr_{lang}.log       the full run log, including progress lines
```

## Reproducing this build

```bash
python -m off_demo_extract.extract \
    --lang en --require-category \
    --input  <dump>.jsonl.gz \
    --taxonomy <taxonomy>/categories.json \
    --output off_en_v14.ndjson \
    --report report_en_v14.json \
    --progress-every 500000 --yes

python scripts/verify_catalog.py \
    --ndjson off_en_v14.ndjson --taxonomy <taxonomy>/categories.json \
    --lang en --json verify_en_v14.json

python scripts/build_manifest.py \
    --dump <dump>.jsonl.gz --taxonomy <taxonomy>/categories.json \
    --locale en:report_en_v14.json:verify_en_v14.json:off_en_v14.ndjson \
    --locale es:... --locale fr:... \
    --out builds/2026-08-03/build_manifest.json
```

`--require-category-path` is on by default and was not overridden. `--taxonomy`
is passed explicitly so the run cannot silently download a newer snapshot than
the one it records.

## What this build does not do

* **No index was written**, and no cluster was contacted. Indexing is a separate
  pass on the maintainer's say-so.
* **The dump is still the January one.** See the trade above.
* Issue #12's third acceptance criterion — comparing an *indexed* vocabulary
  against the pinned snapshot — is half done: every value in these files is
  checked against the snapshot here (0 outside it, all three locales), but the
  comparison against what is actually in an index needs a cluster and belongs
  with the indexing pass. *(Done since, read-only, in
  [`INDEX_VERIFICATION.md`](INDEX_VERIFICATION.md).)*
