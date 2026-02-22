# Mapping Forest Types in Pakistan with NASA PACE

This repository provides a practical starter workflow for using **NASA's PACE mission** observations to map different forest types in Pakistan.

## What this includes

- A reproducible Python script template: `scripts/pace_pakistan_forest_map.py`
- End-to-end workflow notes in this README
- Suggested forest classes and validation approach

## Forest classes you can map

You can adapt class names to your field campaign, but a useful starting taxonomy for Pakistan is:

1. Coniferous montane forest (e.g., Himalayan moist temperate)
2. Dry temperate conifer forest
3. Subtropical broadleaf / chir pine transition
4. Riverine forest
5. Irrigated plantation forest
6. Mangrove forest (Indus delta)
7. Scrub woodland (optional non-forest/transition class)

## Why use PACE for forest mapping?

PACE's Ocean Color Instrument (OCI) provides hyperspectral reflectance. Even though PACE is ocean-focused, atmospheric and land-adjacent signals can still support land-cover discrimination when combined with:

- cloud masking
- terrain correction
- training labels from trusted land-cover/field sources

The script in this repo demonstrates this fused approach.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run:

```bash
python scripts/pace_pakistan_forest_map.py \
  --start-date 2025-01-01 \
  --end-date 2025-03-31 \
  --training-samples data/training_samples.geojson \
  --out-raster outputs/pakistan_forest_types.tif
```

## Inputs expected by the script

- `training_samples.geojson`: polygons or points with attribute `forest_type`
- Internet access and Earthdata credentials configured for NASA data access

## Outputs

- Classified GeoTIFF of forest types over Pakistan
- Console classification report and feature importance

## Validation recommendations

- Hold out independent validation points from each eco-region.
- Report overall accuracy, per-class F1, and confusion matrix.
- Compare seasonal windows (winter vs monsoon) for class separability.

## Notes

This is a starter implementation. You may need to update the exact PACE collection short name(s) or variable names based on the latest Earthdata catalog naming.
