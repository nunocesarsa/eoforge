---
tags: [project]
---
# Roadmap

## Data
1. **STAC source** (`source: stac`): search Earth Search or Planetary Computer by area and dates, cloud-mask with SCL, composite, write the GeoTIFF the rest already reads. `tools/get_tutorial_data.py` already does the search, SCL check and offset handling for single scenes; move that into an [[ImageSource]] subclass.
2. **Earth Engine source** (`source: gee`): export an `ee.Image` (e.g. Sentinel-2 L1C for Dynamic World, or an existing KAZA composite) to GeoTIFF.

## Models
3. **More ready-made models**: Clay embeddings, SatlasPretrain heads; fine-tuning a zoo model on your labels from the app.
4. **Better transfer for trained models**: more LandCoverNet chips, pretrained encoders, early stopping; see [[Test - LandCoverNet pooled run]].
5. **Architecture comparison command**: several configs over repeated spatial splits and held-out scenes, ranked, with a Random Forest baseline.

## Validation and outputs
6. **Area-adjusted accuracy** and unbiased area estimates with confidence intervals (Olofsson et al. 2014).
7. **Multi-temporal inputs** (time series, not only seasonal stacks).
8. **Calibration** of probabilities (e.g. temperature scaling), so entropy can be read as error risk.

## Known limitations
- Whole images are loaded into memory; very large scenes need windowed reading.
- No L2A-to-L1C conversion: Dynamic World and Prithvi floods need L1C.
- TerraMind tiles are generated independently; seams can show at tile edges.
- STAC and GEE sources are stubs.
