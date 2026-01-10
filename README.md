# off-demo-extract

Extract a clean NDJSON catalog (English title + English description + image URL + synthetic price)
from the Open Food Facts JSONL export.

This repository contains **code only**. It does not ship Open Food Facts data.

## Input data

Download the Open Food Facts JSONL export yourself, for example:
- openfoodfacts-products.jsonl.gz / openfoodfacts-products.jsonl (Open Food Facts exports)

Refer to Open Food Facts documentation for data reuse and export details.

## Output format

Each line is a JSON object like:

```json
{
  "id": "0000101209159",
  "title": "...",
  "description": "...",
  "image_url": "https://images.openfoodfacts.org/images/products/000/010/120/9159/front_en.3.400.jpg",
  "price": 4.99,
  "currency": "EUR",
  "brand": "…",
  "categories_tags": ["…"],
  "lang": "en"
}