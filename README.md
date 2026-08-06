# open-food-facts-ndjson-extractor

Extract a clean **NDJSON** demo catalog from the Open Food Facts (OFF) JSONL export.

This repository contains **code only**. It does **not** ship Open Food Facts data or any derived dataset.

More context and rationale: [From messy product feeds to demo-ready e-commerce data](https://alexmarquardt.com/elastic/ecommerce-demo-data/)

## What this produces

This tool transforms raw, complex Open Food Facts data into a flattened, search-ready **NDJSON** format. 

### Schema Overview

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | string | GTIN-13 barcode (padded). |
| `title` | string | Product name (English). |
| `brand` | string | Manufacturer or brand name. |
| `description` | string | Synthesized description (Title + Ingredients + Key Specs). |
| `price` | float | Synthetic, deterministic price for e-commerce simulation. |
| `currency` | string | Currency code (default: USD). |
| `image_url` | string | Computed primary product image URL. |
| `taxonomy_tags` | list | Cleaned, flat list of the product's own category tags, de-duplicated, **validated against the taxonomy** (see [Tags that are not taxonomy categories](#tags-that-are-not-taxonomy-categories)) and rendered with the taxonomy's display name for `--lang` — the **same** label the matching `category_path` segment carries, so the two fields can be joined on string. At most 20 values, and never at the expense of a node on the product's own `category_path` (see [How long the flat list is](#how-long-the-flat-list-is)). |
| `category_path` | list | **Hierarchical** category path — a single root→leaf chain as cumulative `/`-joined strings (e.g. `["Snacks", "Snacks/Salty snacks", "Snacks/Salty snacks/Crisps"]`), reconstructed from the Open Food Facts category taxonomy graph. Powers breadcrumb-style, drill-down category facets. |
| `attrs` | object | **Flattened Dictionary** of key-value attributes (e.g., Nutri-Score, Energy). Most values are strings; the four attributes read from an Open Food Facts list — `Labels`, `Allergens`, `Ingredients analysis`, `Dietary restrictions` — are **lists of values** (see [Multi-valued attributes](#multi-valued-attributes)). |
| `attr_keys` | list | List of all keys available in `attrs` for faceting. |
| `dietary_restrictions` | list | Extracted dietary tags (e.g., vegan, vegetarian). |

## Why this tool is necessary

The raw data from Open Food Facts is incredibly detailed but also complex, containing hundreds of fields, nested objects, and language-specific keys. This makes it difficult to use directly in many applications, especially search and recommendation engines that expect a simple, flat document structure.

This script transforms the raw data into a clean, consistent, and search-ready format. It performs several key operations:

*   **Selects a primary language:** It extracts titles and descriptions from a complex, multi-language structure into single `title` and `description` fields.
*   **Constructs a reliable image URL:** It navigates nested image metadata to build a single, high-quality `image_url`.
*   **Synthesizes a full description:** It combines the title, generic name, and key attributes into a comprehensive `description` field.
*   **Generates a synthetic price:** It creates a deterministic, plausible price to enable e-commerce simulations.
*   **Flattens the structure:** It extracts key attributes into a simple key-value `attrs` object, keeping list-valued attributes as lists so each value stays exactly matchable (see [Multi-valued attributes](#multi-valued-attributes)).
*   **Reconstructs a category hierarchy:** It resolves the product's categories against the Open Food Facts category taxonomy graph to emit a single clean root→leaf `category_path` (see [Category hierarchy](#category-hierarchy)).

### Before: Raw Open Food Facts Data Example

A single product can have over 500 fields. Key information like the product name or image URL is buried in nested objects and requires logic to extract.

```json
{
  "_id": "0008127000019",
  "pnns_groups_2": "Fats",
  "labels_old": "",
  "ingredients_from_palm_oil_tags": [],
  "brands": "Athena Imports",
  "code": "0008127000019",
  "editors_tags": [
    "aleene",
    "clockwerx",
    "ecoscore-impact-estimator",
    "kiliweb",
    "usda-ndb-import",
    "yuka.sY2b0xO6T85zoF3NwEKvlkBmTtT-iz2cKD3tvnWnxIyJDIfrfuxX2KLROas",
    "yuka.sY2b0xO6T85zoF3NwEKvlnAfXOTz-RmeOxLgh3egzemsPJb1YepV7aSnHas"
  ],
  "packaging_materials_tags": [],
  "ingredients_sweeteners_n": 0,
  "generic_name": "",
  "categories_properties": {
    "agribalyse_food_code:en": "17270",
    "ciqual_food_code:en": "17270",
    "agribalyse_proxy_food_code:en": "17270"
  },
  "nutriscore_2021_tags": [
    "c"
  ],
  "languages_tags": [
    "en:english",
    "en:1"
  ],
  "link": "",
  "purchase_places": "",
  "ingredients_with_unspecified_percent_sum": 100,
  "ingredients_with_specified_percent_n": 0,
  "origin_en": "",
  "nutriscore_version": "2023",
  "...": "hundreds of additional fields omitted"
}
```

### After: Cleaned NDJSON for Search Example

The output is a clean, flat JSON object, ready to be indexed into a search engine like Elasticsearch or OpenSearch.

```json
{
  "id": "0008127000019",
  "title": "Extra virgin olive oil",
  "brand": "Athena Imports",
  "description": "Extra virgin olive oil. Extra virgin olive oil Key specifications: Category: Plant-based foods and beverages; Serving size: 15 ml; Nutri-Score: B; NOVA group: 2; Eco-Score: E; Dietary restrictions: vegan, vegetarian; Ingredients analysis: palm-oil-free, vegan, vegetarian; Energy (kcal/100g): 800 kcal; Fat (g/100g): 93.3 g; Saturated fat (g/100g): 13.3 g; Sugars (g/100g): 0 g; Salt (g/100g): 0 g; Protein (g/100g): 0 g; Countries: United States",
  "image_url": "https://images.openfoodfacts.org/images/products/000/812/700/0019/front_en.5.400.jpg",
  "price": 14.29,
  "margin": 22,
  "popularity": 0,
  "currency": "USD",
  "taxonomy_tags": [
    "Plant-based foods and beverages",
    "Plant-based foods",
    "Fats",
    "Vegetable fats",
    "Olive tree products",
    "Vegetable oils",
    "Olive oils",
    "Extra-virgin olive oils",
    "Virgin olive oils"
  ],
  "category_path": [
    "Plant-based foods and beverages",
    "Plant-based foods and beverages/Plant-based foods",
    "Plant-based foods and beverages/Plant-based foods/Olive tree products",
    "Plant-based foods and beverages/Plant-based foods/Olive tree products/Olive oils",
    "Plant-based foods and beverages/Plant-based foods/Olive tree products/Olive oils/Virgin olive oils",
    "Plant-based foods and beverages/Plant-based foods/Olive tree products/Olive oils/Virgin olive oils/Extra-virgin olive oils"
  ],
  "attrs": {
    "Serving size": "15 ml",
    "Nutri-Score": "B",
    "NOVA group": "2",
    "Eco-Score": "E",
    "Ingredients analysis": [
      "palm-oil-free",
      "vegan",
      "vegetarian"
    ],
    "Countries": "United States",
    "Category": "Plant-based foods and beverages",
    "Energy (kcal/100g)": "800 kcal",
    "Fat (g/100g)": "93.3 g",
    "Saturated fat (g/100g)": "13.3 g",
    "Sugars (g/100g)": "0 g",
    "Salt (g/100g)": "0 g",
    "Protein (g/100g)": "0 g",
    "Dietary restrictions": [
      "vegan",
      "vegetarian"
    ],
    "Price source": "estimated_unit_model",
    "Pricing bucket": "olive_oil",
    "Estimated unit price": "28.68 USD/l (default 500ml (no package qty), source=none, bucket=olive_oil)",
    "Margin source": "modelled_category_margin",
    "Modelled margin": "22% (bucket=olive_oil base=22%)",
    "Popularity source": "open_food_facts_unique_scans",
    "Unique scans (Open Food Facts)": "0"
  },
  "attr_keys": [
    "Category",
    "Countries",
    "Dietary restrictions",
    "Eco-Score",
    "Energy (kcal/100g)",
    "Estimated unit price",
    "Fat (g/100g)",
    "Ingredients analysis",
    "Margin source",
    "Modelled margin",
    "NOVA group",
    "Nutri-Score",
    "Popularity source",
    "Price source",
    "Pricing bucket",
    "Protein (g/100g)",
    "Salt (g/100g)",
    "Saturated fat (g/100g)",
    "Serving size",
    "Sugars (g/100g)",
    "Unique scans (Open Food Facts)"
  ],
  "dietary_restrictions": [
    "vegan",
    "vegetarian"
  ]
}
```

### Multi-valued attributes

`attrs` is intended to be indexed as an Elasticsearch `flattened` field, and
`flattened` indexes **each element of an array** as its own keyword. So the
shape a value is written in decides whether it can be queried:

| written as | `{"term": {"attrs.Labels": "no-gluten"}}` | `terms` aggregation |
| :--- | :--- | :--- |
| `["no-gluten", "vegetarian", "green-dot"]` | matches | three buckets, one per value |
| `"no-gluten, vegetarian, green-dot"` | no match | one bucket: the whole string |

Four attributes are read from an Open Food Facts **list** field and are
therefore written as lists:

| attribute | Open Food Facts source |
| :--- | :--- |
| `Labels` | `labels_tags` |
| `Allergens` | `allergens_tags` |
| `Ingredients analysis` | `ingredients_analysis_tags` |
| `Dietary restrictions` | derived from `labels_tags` + `ingredients_analysis_tags` |

Every other attribute is a single value and stays a string. **That distinction
cannot be recovered from the value itself**: `Modelled margin`, `Estimated unit
price`, `Serving size` and `Quantity` all carry commas *inside* one legitimate
value — `"10x 23g (5x 46 g), Net: 230 g"` is one quantity, not two. Splitting
`attrs` values on `", "` anywhere downstream shreds them. The list/scalar
distinction is made once, at the writer, where the source is still a list.

`Countries` is the attribute that looks like a fifth and is not one. It is read
from the free-text `countries` field, which is prose written by whoever edited
the product — the dump carries `"France, United States"`,
`"Frankreich,Deutschland"` and `"France,États-Unis,en:france"` — so there is no
list at this call site to preserve. Open Food Facts publishes the canonical list
separately as `countries_tags`; reading that instead would change both the source
field and the displayed value (`United States` → `united-states`), which is a
decision about the catalog rather than a join to undo. Tracked in
[#50](https://github.com/alexander-marquardt/open-food-facts-ndjson-extractor/issues/50).

Display is unaffected: the generated `description` joins list attributes with
`", "` at render time, exactly as the values used to be joined at write time, so
every description is byte-identical to the one the joined form produced.

## Category hierarchy

Open Food Facts ships `categories_tags` (and an identical `categories_hierarchy`)
on every product, but **those are not a single path** — they are the flattened
*union of every ancestor category* drawn from the Open Food Facts category
taxonomy, which is a directed acyclic graph (a category can have several
parents). Naively joining the tags with `/` mixes parallel roots and sibling
branches and yields a nonsense path.

To produce a real tree — the same cumulative-path shape merchandising tools use
for drill-down facets — the extractor loads the public OFF category taxonomy and
walks a single canonical chain:

1. **Once per run**, collapse the whole taxonomy DAG to a spanning forest by
   giving every category one canonical parent (see [Canonical parents and the
   tie-break](#canonical-parents-and-the-tie-break)). The whole taxonomy, in
   every language — the same forest for every catalog.
2. Keep the product's tags that exist in the taxonomy (drops noise and
   foreign-language-only nodes for the target language).
3. Pick the most specific of those tags as the leaf, and walk the canonical
   parent map from it to a **global** taxonomy root — materialising ancestors the
   product never tagged, in whatever language the taxonomy files them under.
4. Emit cumulative `/`-joined path strings using the taxonomy's localized display
   names.

Note where the target language does and does not apply. It decides step 2 —
which categories a product may be *filed under*, and which values the flat
`taxonomy_tags` field may carry — and it decides the labels in step 4. It does not
decide step 1. Nor does it reach `xx:`, which is not a language but upstream's
marker for a node whose name is the same in every language: those 34 nodes are
filable in **every** catalog.

Filtering the *graph* by language used to delete every edge
through a filtered node as well, which orphaned that node's children into roots
of a locale-shaped forest: an English run walked a graph with **161** roots
where the taxonomy has **92**, a Spanish one 161 and a French one 130, so the
same category sat at a different depth in each catalog. Now one language-blind
graph serves all three, and a chain crosses a foreign ancestor instead of being
severed at it — `en:pate` reaches `Meats and their products/Prepared meats/
Charcuteries diverses/Pâté` rather than being filed as a top-level `Pâté`.

The visible cost is that one segment: of the 8,965 nodes an English path may
contain, 26 are foreign and 20 of those have no English or `xx` name, so they
render de-slugged from their own language. It applies to 0.9% of the categories
an English catalog can file under, and it buys back their true lineage.

```text
raw tags:   en:plant-based-foods-and-beverages, en:beverages, en:hot-beverages,
            en:plant-based-beverages, en:teas, en:tea-bags        (a flat DAG union)

category_path:
  [ "Beverages",
    "Beverages/Hot beverages",
    "Beverages/Hot beverages/Teas" ]                              (one clean chain)
```

### Canonical parents and the tie-break

Step 1 is what makes the hierarchy usable for faceting, and it is the reason the
walk is anchored globally rather than to the product's own tags. Two properties
have to hold, and both are covered by tests in `tests/test_taxonomy.py`:

- **One address per category.** A category occupies exactly one position in the
  tree, so every product carrying it files it at the same path. Anchoring only to
  the product's own tags broke this whenever a product omitted an intermediate
  tag: the walk stopped at whatever the product happened to hold and invented a
  shorter path, so the same category showed up at two addresses and its facet
  count split in two.
- **One path per product.** Each product carries exactly one root→leaf chain,
  never a union of branches.

The canonical parent of each node is chosen by **fewest hops to a taxonomy root**,
and **on a tie the lexicographically smallest canonical id wins**. The tie-break
is not a corner case: 2,545 of the 14,457 categories have several parents and
1,070 of those tie on depth, so it decides nearly half the multi-parent cases.

It has to be *stable across taxonomy refreshes* — if a category's address moved
between rebuilds, previously authored merchandising rules would silently stop
matching. Lexicographic order depends only on the set of tied ids, so upstream
re-ordering a `parents` list cannot move an address. On the current taxonomy all
2,545 multi-parent nodes already list their parents in lexicographic order, so
this rule agrees with the upstream authored order on every tie today, while not
being hostage to it.

Collapsing the DAG this way cuts 2,769 of the taxonomy's 17,134 parent edges and
orphans **zero** categories: every non-root keeps exactly one parent and all 92
roots stay roots. Only redundant parent relationships are dropped.

Each run's report carries a `category_path_addresses` block counting the
categories that resolved to more than one address, the categories that rendered
under more than one label, and the labels claimed by more than one category. All
three should always read zero; if any does not, the extraction log says so.

### One label per category

The flat `taxonomy_tags` list is still emitted alongside the hierarchical
`category_path` (it also drives pricing-bucket matching and the `attrs.Category`
field). Both are derived from the product's `categories_tags`, and **both take a
category's label from one function** — `display_label` in
`src/off_demo_extract/taxonomy.py` — so a change to how categories are named
cannot reach one field and not the other.

That label is the taxonomy's `name` for `--lang`, falling back to English, then
`xx`, then a prettified slug. It is preferred over de-slugging the tag id because
it is the upstream-authored human label: it carries correct hyphenation
(`Plant-based foods`), disambiguating parentheticals (`Crackers (Appetizers)`)
and casing that a mechanical de-slug destroys — and, unlike a slug, it is
localized. All 8,939 English-backbone nodes have an English name, so the slug
fallback never fires on that backbone.

One thing is changed about that name and only one: **its first character is
upper-cased when it is a lowercase letter**, and no other character is touched.
Upstream does not capitalize consistently — 92 of the 8,939 English-backbone
names begin lowercase (`ice creams`, `chorizo`, `baker's yeast`), as do 49
Spanish and 208 French names — so a real product reads
`Desserts/Frozen desserts/Ice creams and sorbets/ice creams/Ice cream tubs`
mid-breadcrumb and carries the same uncapitalized value in its `taxonomy_tags`
row, since both fields render from the same labeller. It is not `str.capitalize()`, which would flatten the rest of
the string and cost `dried Toothed wrack` its species capital; a name starting
with a capital, a digit or a punctuation mark (`10% red wine`,
`% de matières grasses`) is returned byte-identical. The slug fallback
already capitalized its first character, so the same node used to render
`Saint-émilion` when the taxonomy had no name for it and `saint-émilion` when it
did; one helper now decides the casing for both. All 92 were read before the rule
was chosen, and none is deliberately lowercase; they are kept verbatim in
`tests/fixtures/off_real_lowercase_names.json` so a taxonomy refresh is checked
against the same question. **The source file is not modified** — this is how a
label is presented, not what upstream says.

`taxonomy_tags` used to de-slug the tag id itself. That gave the same node two
spellings across the two fields (`Plant based foods` next to `Plant-based
foods`), so nothing could relate a flat value to a path segment by string: over
the first 200,000 lines of the public export, only 75.1% of products had every
self-tagged chain node's label present verbatim in `taxonomy_tags`; it is now
100% — the last 0.002% being the three products the flat field's length cap used
to truncate, described in [How long the flat list is](#how-long-the-flat-list-is).
Because the de-slug worked off the `en:`-prefixed tag id, it also emitted English
labels in **every** locale — a Spanish catalog rendered `Plant based foods` in
`taxonomy_tags` and `Alimentos de origen vegetal` in `category_path`, in the same
document. Both fields are now localized together.

Where the taxonomy has no translation the label falls back to English in both
fields, which is a gap in the upstream taxonomy rather than a disagreement
between the fields: of the 8,939 English-backbone nodes, 86.0% have a French name
and 34.6% a Spanish one.

Note that the two fields are *relatable*, not identical: `category_path` is
anchored to a global taxonomy root, so it materialises ancestors the product
never tagged, and those ancestors are legitimately absent from `taxonomy_tags`.
`tests/test_category_label_agreement.py` pins the direction that must hold —
every chain node the product *did* tag appears verbatim in `taxonomy_tags` — by
exact string, on real Open Food Facts records.

### How long the flat list is

`taxonomy_tags` carries at most 20 values. The cap is not a storage rule: the
field was introduced carrying 3 values and raised to 20 in "Increased the number
of categories extracted" with no reason recorded, and nothing downstream reads a
length — PRISM maps the field as a `terms` facet, which has none. It is kept
because Open Food Facts occasionally tags a product very heavily (33 tags is the
most in the first million records) and an unbounded field has no ceiling at all,
not because a measured cost forces one: removing it outright would add 205 bytes
to 10.5 MB of emitted tag payload, +0.002%.

What the cap must not do is drop a tag the product's own `category_path` needs.
It used to. The cap was applied by walking the tags in order and stopping, which
drops the *tail* — and Open Food Facts orders `categories_tags` roughly
general-to-specific, so the tail is where a product's most specific tags are. Over
the first 200,000 records of the public export, 6 of 135,716 tagged products
have more eligible tags than the cap and 3 of those lost a node of their own
chain to it: `0036800388352` its `Basmati rices`, `0051933012707` and
`0078742086774` their `Peas` — a `category_path` segment with no flat
counterpart, which reads exactly like the labelling divergence fixed above and is
not one.

So the product's chain is reserved: every tag on it survives, and the cap governs
the remaining, *incidental* tags. That makes the invariant hold by construction
rather than by there happening to be few enough tags — raising the cap to 24
would have covered the longest list in that sample and left the same defect
waiting for the 25-tag product two paragraphs up. Selection changes; **order does
not**, so `taxonomy_tags[0]` is still the primary tag's label, which `attrs` and
the generated description read back. On the 15,687-record English catalog that
those 200,000 lines produce, the fix changes 2 documents (both gain `Peas` in
place of `Cooked garden peas`) and the file shrinks by 28 bytes.

A chain longer than the cap would take the list past 20 rather than lose a
segment. None exists today — the deepest chain in that sample is 9 nodes — and
the run report counts the case so a deeper taxonomy says so rather than
silently reopening the defect.

Every run reports what the cap discarded, in a `taxonomy_tags_cap` block: the
number of products truncated, the number of labels dropped, the labels
themselves, and the longest eligible list seen. A tag dropped here is *valid* —
it survived curation — so it appears nowhere else in the report, and the emitted
field has no marker distinguishing a shortened list from a short one.
`tests/test_category_tag_cap.py` pins both halves on real records: no chain node
is ever the casualty, and the incidental tags past the cap are still dropped.

### Tags that are not taxonomy categories

A product's `categories_tags` are **not** guaranteed to be taxonomy nodes. Legacy
tags, contributor-entered tags and tags upstream has since renamed all ride along
in the export. Measured over the first 300,000 records of the January 2026 dump
against the pinned 14,457-node snapshot, after the `en:undefined`/`en:null`
sentinels `--category-exclude` already removes:

| | |
| :--- | ---: |
| products with a real category tag | 218,239 |
| tag instances | 1,171,851 |
| instances with no taxonomy node | 49,962 (4.26%) |
| distinct such tags | 7,712 |
| products carrying at least one | 42,727 (19.6%) |
| products where *every* tag is one | 5,456 (2.50%) |

Those tags used to be rendered into the searchable `taxonomy_tags` field as if they
were canonical. `Groceries` alone was searchable on 6,299 documents of the built
English catalog — a value with no node in the hierarchy, so it can never be a
`category_path` segment and can never be authored as a merchandising rule value.
It is also a value the labeller has no `name` to render, only a slug.

Each tag now goes through, in order:

1. **A curated alias map** — `TAG_ALIASES` in
   `src/off_demo_extract/category_tags.py`, an id upstream retired mapped to the
   node that *is* that category today (`en:easter-food` →
   `en:easter-foods-and-drinks`). Most entries are derived from upstream's own
   synonym lines, which is where a rename leaves its trace; the two that are not
   are marked inline. Aliases are canonicalizations, never generalizations: a tag
   is never filed under a *broader* node, because that would put a product in a
   category it never claimed.
2. **A curated drop list** — `TAG_DROPS`, each entry carrying the reason it is
   refused. `en:groceries` is 36.6% of all unresolvable instances on its own and
   is contentless in a grocery catalog; `en:aoc-cheeses` and `en:labeled-cheeses`
   are label *attributes* rather than places in a product hierarchy.
3. **Membership in the run's category vocabulary** — the same node set a product
   may be *filed under*, so the flat field and the tip of the path cannot draw on
   different vocabularies. A tag that is a real node but in a language this
   catalog does not file under (`fr:charcuteries-cuites` in an English catalog)
   is refused here too: no English product is filed under it either, so leaving
   it in the flat field produces a value that is searchable but never facetable.
   The same node can still appear as an intermediate *path segment*, because a
   path walks the language-blind graph — an English product filed under
   `en:poultry-hams` passes through `fr:charcuteries-cuites`, which is where that
   category genuinely sits. `xx:` is the exception, and is not a language: it
   marks the 34 nodes upstream names identically in every language (`xx:tofu`,
   `xx:dumplings`, `xx:sake`), so every catalog files under them.

**A refused value never refuses its record.** Dropping the record on an
unresolvable tag would cost 19.6% of tagged products; dropping only the offending
value costs 2.50%, because in 87% of cases the product has a complete valid
lineage and one junk tag riding along. The 2.50% that end up with nothing are
exactly the products `--require-category-path` already drops.

Every run reports what it refused, in a `category_tag_curation` block: totals by
reason, the distinct unresolvable-tag count, the rate, and the worst offenders by
name. The rate is meant to be read per run — this defect was originally found by
reverse-mapping a built index against the taxonomy, which nobody should have to
do twice.

The taxonomy file is fetched once from
[`https://static.openfoodfacts.org/data/taxonomies/categories.json`](https://static.openfoodfacts.org/data/taxonomies/categories.json)
and cached at `data/taxonomy/categories.json`. Override its location with
`--taxonomy <path>`. Display names follow `--lang`, so the French and Spanish
personas get localized category labels — in `taxonomy_tags` as well as in
`category_path`.

### Category-hierarchy flags

Two separate flags control the hierarchy. They are easy to confuse, so here is
exactly what each does:

- **`--no-taxonomy`** — turns the hierarchy feature *off entirely*. The extractor
  does **not** load (or download) the taxonomy, and every product is written with
  an empty `category_path` (`[]`). The flat `taxonomy_tags` field is still written,
  but with no taxonomy there are no display names to read, so its labels fall
  back to a prettified slug of the tag id — English, and without the taxonomy's
  hyphenation or parentheticals. With no taxonomy there is also no vocabulary to
  validate tags against, so only the curated drop list applies and the rest are
  emitted unchecked; refusing everything would blank the field for the whole run.
  Use this to skip the download when you don't need the hierarchy and are not
  relying on category labels.
- **`--require-category-path`** *(default: on)* — **drops products whose
  `category_path` can't be reconstructed**, so the catalog is uniformly
  drill-down-faceted. This is a small tail — roughly 2–6% of otherwise-clean
  records in the EN/FR/ES full runs. Pass `--no-require-category-path` to keep
  those products with an empty `category_path` instead.

  "Reconstructed" means **anchored**, not merely non-empty: the chain has to
  reach one of the taxonomy's 92 global roots, not just start somewhere. A path
  that starts mid-taxonomy files its categories at an address no other catalog
  agrees with, and it is non-empty and well-formed, so an emptiness test cannot
  see it.

  On a default run this refuses nothing — the graph the chain walks is the whole
  taxonomy, so its roots *are* the 92. (Before traversal went language-blind,
  the language filter severed 455 chains per 300,000 records of the January 2026
  dump for English and 2 for French, 6 of which also cleared the
  title/description/image filters and were dropped.) What can still strand a
  chain is `--category-exclude` naming a mid-taxonomy node, which orphans
  everything beneath it. Every refusal is named in the report's
  `category_path_anchoring` block, which is populated whether or not the gate is
  on — the standing zero is what shows the forest still has exactly the
  taxonomy's roots.

The interaction is the part to watch: with `--no-taxonomy` *every* path is empty,
so naively "requiring a path" would drop **every** product and produce an empty
file. To prevent that, **`--no-taxonomy` automatically disables the
require-path gate** (a log line notes this). The same graceful handling applies
if the taxonomy simply fails to load (e.g. no network on first run): paths come
out empty, the gate is disabled, and the run continues rather than crashing or
silently emitting an empty dataset.

| Flags | `category_path` | Products dropped for an unresolved path? |
| :--- | :--- | :--- |
| *(default)* | populated from the taxonomy | **Yes** — the ~2–6% empty tail, plus anything `--category-exclude` stranded |
| `--no-require-category-path` | populated from the taxonomy | No (both are still counted in the report) |
| `--no-taxonomy` | always `[]` | No (gate auto-disabled) |
| taxonomy fails to load | always `[]` | No (gate auto-disabled, with a warning) |

## Verifying a build

A catalog passes through three artifacts, and each one is checked against the
one before it. The chain matters: a claim about an index that rests only on the
code that wrote it is not a check.

| Stage | Script | Reads | Answers |
| :--- | :--- | :--- | :--- |
| Artifact | `scripts/verify_catalog.py` | the NDJSON + the taxonomy snapshot | is the file tree-shaped, anchored, and inside the snapshot's vocabulary? |
| Identity | `scripts/build_manifest.py` | the dump, the taxonomy, the run reports | what was this built *from*, by checksum? |
| Index | `scripts/verify_index.py` | a live index + the manifest | does what got loaded match what was built? |

The third stage is the one that used to be missing, and its absence cost
something concrete: `catalog_fr_v13` held 195,209 documents where its extract has
222,955 distinct ids, and the 12.5% shortfall survived months and three index
generations because no check ever put the two numbers side by side.

### Checking the catalog on disk

```bash
python scripts/verify_catalog.py \
    --ndjson data/products/off_en_v14.ndjson \
    --taxonomy data/taxonomy/categories.json \
    --lang en --json builds/2026-08-03/verify_en_v14.json
```

stdout is the JSON result and nothing else, so a build can capture it straight to
a file; the summary, the tolerance in force and any failure reason go to stderr.

| Exit | Meaning |
| :--- | :--- |
| 0 | every property holds (within any tolerance named on the command line) |
| 1 | the catalog fails a gate — every failing one is named in `failed` and on stderr |
| 2 | the verification could not be carried out: a file is missing, or a line is not a JSON object |

**Every property it measures is fatal, at zero tolerance.** That is a change:
`values_outside_pinned_snapshot` used to be measured and printed and then left
out of the sum that became the exit status, so a catalog none of whose values the
pinned snapshot explains exited 0 — while `verify_index.py` treats the same
condition on the index side as a failure, leaving the two ends of one rule
disagreeing about whether it was fatal.

An exception is named rather than budgeted for: `--allow-values-outside-snapshot
N`, or `--allow-values-outside-snapshot-fraction F` of the distinct values
actually checked. The fraction is **floored** to a whole number of values, so it
can never round up into permitting one more, and the tolerance in force is
recorded in the JSON (`values_outside_snapshot_tolerance` and
`..._tolerance_source`) rather than only printed — a record that says
"0 values outside the snapshot" is worth what the tolerance beside it says.

Two further rules follow from the same reasoning:

* **A run that verified nothing does not report clean.** An empty catalog, or one
  in which no record carries a `taxonomy_tags` value or a `category_path` segment,
  passes every count-based gate vacuously. It is also the denominator a fraction
  tolerance divides by.
* **Duplicate ids are reported, and fatal only under `--require-unique-ids`.**
  Two records sharing an id is a property of the upstream dump, not of anything
  the extractor constructs, and the index is keyed by id, so the duplicate
  resolves to a single document — which is exactly why `verify_index.py` compares
  `_count` against `distinct_ids` rather than against the record count. The
  duplicates are accounted for downstream rather than unexplained. The 2026-08-03
  build carries 1 / 3 / 81 of them in en / es / fr, and all three pass.

```bash
export PRISM_ELASTICSEARCH_URL=...  PRISM_ELASTICSEARCH_API_KEY=...

python scripts/verify_index.py \
    --index catalog_fr_v13 \
    --manifest builds/2026-08-03/build_manifest.json \
    --taxonomy data/taxonomy/categories.json
```

It compares the index's document count against the manifest's **distinct ids**
(not its record count — an index keyed by id holds one document per id, and the
two differ by 1/3/81 for en/es/fr), reports every `taxonomy_tags` value and
`category_path` segment the pinned snapshot does not explain *and* every snapshot
label the index never uses, and says whether the index records which build
produced it. Today no index does, so that last check reports `unverifiable`
rather than pretending the manifest on the command line was confirmed by
anything; it prints the `_meta` block a loader would have to write to make the
answer real.

**A check that read nothing does not report clean here either.** This is the
same rule as the catalog side's, applied to the same class of gate: every check
here counts things that are wrong, and a count over an empty read is zero, so a
check handed nothing used to report `pass`. It is not hypothetical — against
`catalog_en_v14` before the `taxonomy_tags` rename, the flat half of
`category_vocabulary` read 0 distinct values, found 0 outside the snapshot,
reported all 14,453 snapshot labels unused, and still passed, hiding 46 real
vocabulary defects that appeared the moment the field name was corrected. A
`terms` aggregation on a field the mapping does not have is not an error:
Elasticsearch answers with an empty bucket list, `sum_other_doc_count: 0`,
`doc_count_error_upper_bound: 0` and `_shards.failed: 0`, and no signal in that
response distinguishes it from a field the index genuinely holds no values of.

So the mapping — which this script reads anyway — now confirms the fields the run
is about to aggregate on before a single bucket is counted (`mapped_fields`), and
an empty vocabulary read is a failure that names which of the two it was: the
mapping does not declare the field (a **blind** read, fix the verifier), or it
declares it and the index holds no value of it (a **legitimately empty** read,
fix the index). `snapshot_labels_unused_by_index == snapshot_labels` — "the index
uses none of the taxonomy" — stays informational, because it is now reachable
only through one of those two states, and both are already fatal.

Adding `--catalog <ndjson>` diffs the two id sets exactly and profiles the
missing ids by run length in catalog order, which is what separates a load that
dropped batches from two extracts that merely disagree record by record. It is
opt-in because it costs a pass over the NDJSON; the default run is two requests.

The whole thing is read-only by construction — one helper refuses any endpoint
outside `_search`, `_count`, `_mapping` and `_settings` — so it is safe to point
at a cluster other people are using. Results for the current indices are in
[`builds/2026-08-03/INDEX_VERIFICATION.md`](builds/2026-08-03/INDEX_VERIFICATION.md).

### Adding `category_path` to an index that is already loaded

`scripts/inject_category_path.py` sends partial `_update` operations keyed by
`_id`, which merge the one field without running the ingest pipeline — so the
embedding vectors already in `_source` are left alone.

A partial update **cannot create a document**. Elasticsearch answers an update
against an id the index does not hold with a per-item `404`, and the bulk request
itself still succeeds — so the interesting output of this tool is not what it
printed, it is what it exited with:

```bash
python scripts/inject_category_path.py \
    --index catalog_en_v14 \
    --ndjson data/products/off_en_v14.ndjson
```

| Exit | Meaning |
| :--- | :--- |
| 0 | every document sent was applied, or the misses stayed inside a tolerance that was named on the command line |
| 1 | the run completed and its outcomes fail the gate |
| 2 | the run could not complete — the bulk request itself failed |

Every per-item outcome is counted apart from the others (`updated`, `noop`,
`not_found`, `conflict`, `failed`, and `unaccounted` for documents the response
said nothing about), the applied rate is printed, and the first few ids the index
does not hold are named — enough to tell "wrong index" from "stale id set"
without a second query. The default is zero tolerance for a miss: the extract
that built an index has no legitimate reason to address a document that index
does not hold. `--allow-missing N` or `--allow-missing-fraction F` makes an
exception explicit, and the tolerance in force is printed in the report. What no
tolerance can authorise is a run that applied **nothing** — that is what pointing
at the wrong index looks like, and it always fails.

### Copying an index to a new generation

`scripts/reindex_v7_to_v8.py` copies a catalog index server-side with **no
ingest pipeline**, so the embedding vectors already in `_source` are carried
over verbatim instead of being recomputed, and adds the `category_path` field to
the new mapping.

The copy is asynchronous: the cluster answers the `_reindex` with a task id and
does the work afterwards. So this tool has the same honesty problem as the
injector above — it can report success for work it has not seen happen.

```bash
python scripts/reindex_v7_to_v8.py --source catalog_en_v7 --dest catalog_en_v8
```

| Exit | Meaning |
| :--- | :--- |
| 0 | the copy completed and the destination holds the source's documents (within any tolerance named on the command line) |
| 1 | the copy completed and its outcome fails a gate |
| 2 | the run could not complete — a request failed, or the destination exists and `--recreate` was not given |
| 3 | `--no-wait` only: the task was started and **nothing about it has been verified** |

The fourth status is what `--no-wait` is for. Firing a long reindex and polling
separately is a legitimate thing to want, so the flag stays — but a run that has
only submitted a task knows nothing about the copy, and returning `0` for it
made "submitted" indistinguishable from "copied and counted" to an `&&` chain.
`3` is neither a success nor an error: it says the verification is owed, and the
run prints and records the task id to discharge it with.

A source holding **0 documents** fails before the destination is created.
`dst_count == src_count` is satisfied by `0 == 0`, so an empty or misnamed source
used to report a clean copy — the same vacuous pass the injector's "a run that
sent nothing fails" rule closes. `--allow-empty-source` names the exception when
an empty index really is the intended input.

Everything the finished task reports is read and named, including
`version_conflicts`: with `op_type: create` into a freshly created destination a
conflict means the id was already there, so each one is a document that was not
copied. Misses are gated at zero by default, with `--allow-missing N` /
`--allow-missing-fraction F` (floored) as the named exception and the tolerance
in force printed with the result; `--json` writes the whole record, including
whether the copy was verified at all. What no tolerance can authorise is a
non-empty source whose destination ends up holding nothing.

## Clean data definition

- Title in target language: `product_name_{lang}` OR (`lang == {lang}` AND `product_name`) — controlled by `--lang`
- Description in target language: `generic_name_{lang}` OR `ingredients_text_{lang}` OR (`lang == {lang}` AND `generic_name`/`ingredients_text`)
- A front image matching the target language: `images.front_{lang}`
- If `--require-category` is enabled: at least one meaningful category (placeholder/empty categories excluded)
- By default, a `category_path` that resolves against the OFF taxonomy **and reaches one of its 92 global roots** — products whose hierarchy can't be reconstructed, and products whose chain starts mid-taxonomy because `--category-exclude` stranded its ancestry, are both excluded (disable with `--no-require-category-path`)
- Synthetic deterministic price (see Price Estimation)

## A small sample of cleaned NDJSON files

You can look into `data/sample-data` to see a few JSON product documents that were generated by the scripts in this repository

## Example search results with the cleaned data
![Demo screenshot](assets/cleaned-data.png)

## Data is not included in this repo

Users must download the OFF JSONL export at: [https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz](https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz)

Then place it in the data directory:

`data/json_source/openfoodfacts-products.jsonl.gz`

The script reads `.jsonl` or `.jsonl.gz` and streams it; it does not require full decompression.

## Price Estimation

The pricing information is synthetically generated by the `extract.py` script and is not a separate tool. The price estimation logic is based on a product's category, quantity, and other attributes, using the configuration from `config/pricing_buckets.json`. This process is integrated into the main extraction pipeline.

## Project Structure

```text
ecommerce-open-food-facts/
├── data/                  # All data lives here (ignored by git)
│   ├── json_source/       # Raw OFF data dump
│   │   └── openfoodfacts-products.jsonl.gz
│   ├── taxonomy/          # Cached OFF category taxonomy (auto-downloaded)
│   │   └── categories.json
│   └── products/          # Processed NDJSON files (The "Output")
├── src/
│   └── off_demo_extract/  # Python Package
│       ├── extract.py     # Main extraction pipeline
│       ├── taxonomy.py    # OFF taxonomy graph → hierarchical category_path
│       ├── category_tags.py  # Curated aliases/drops + validation of product tags
│       └── pricing.py     # Synthetic price model
├── scripts/
│   ├── verify_catalog.py  # Re-derives the catalog's properties from the NDJSON,
│   │                      #   and fails on every one that does not hold
│   ├── build_manifest.py  # Pins dump / taxonomy / commit by checksum
│   ├── verify_index.py    # Checks a live index against that manifest (read-only)
│   ├── inject_category_path.py  # Adds category_path to an already-loaded index,
│   │                            #   and fails when the updates miss their documents
│   └── reindex_v7_to_v8.py  # Copies an index to a new generation without
│                            #   re-embedding, and reports success only for a
│                            #   copy it has polled to completion and counted
├── builds/                # Per-build manifests, reports and verification notes
├── tests/
│   ├── conftest.py        # Puts src/ on sys.path so tests import normally
│   ├── test_business_signal_values.py  # margin / popularity derivation
│   ├── test_canonical_parents.py       # Canonical parent map and its tie-break
│   ├── test_category_addressing.py     # One address per category, one path per product
│   ├── test_category_path_gate.py      # End-to-end tests for the category_path gate
│   ├── test_index_verification.py      # Index-vs-manifest checks, on captured ES envelopes
│   ├── test_taxonomy.py   # Regression tests for the hierarchy builder
│   └── fixtures/          # Real OFF products + pruned taxonomy, checked in
├── pyproject.toml         # Dependencies
└── README.md
```

### Running the tests

The tests need no network and no data dump — they build a synthetic taxonomy and
a two-record input on the fly:

```bash
uv run --with pytest python -m pytest tests/ -v
```

Each test file is also runnable on its own (`python tests/test_taxonomy.py`).
CI runs the same suite on every push and pull request.

## Quickstart (uv)

Create and run a small sample (recommended first step):

```bash
uv run -m off_demo_extract.extract \
  --lang en \
  --require-category \
  --output data/products/sample_2k_front_en.ndjson \
  --report data/products/sample_2k_front_en_report.json \
  --max-output-records 2000 \
  --max-input-lines 3000000
```

The script will prompt for confirmation if the output file already exists. To bypass this, add the `--yes` flag.

Run full extraction (no arbitrary cutoff; reads to EOF):

```bash
uv run -m off_demo_extract.extract \
  --lang en \
  --require-category \
  --output data/products/off_en_clean_categorized.ndjson \
  --report data/products/report_categorized.json \
  --progress-every 500000
```


## Licensing and data reuse (important)

- This repository’s code is licensed under the MIT License (see `LICENSE`).
- Open Food Facts data and images are governed by Open Food Facts’ licenses and terms.

OFF data reuse is based on ODbL (attribution + share-alike). Image URL construction follows OFF documentation.

See `DATA_LICENSE.md` for links to the authoritative Open Food Facts pages and a short summary.
