---
tags: [module, data]
---
# eoforge.data.sources

Imagery sources. Each source turns a `DataConfig` into a local GeoTIFF; the rest of the pipeline only reads that file. Adding a source therefore does not touch training or prediction.

| Name | Class | Status |
|---|---|---|
| `local` | `LocalGeoTIFF` | Working: checks the file exists and returns its path |
| `stac` | `STACSource` | Stub, raises `NotImplementedError` ([[Roadmap]]) |
| `gee` | `EarthEngineSource` | Stub, raises `NotImplementedError` ([[Roadmap]]) |

**Functions:** [[ImageSource]] · [[register_source]] · [[get_source]]
