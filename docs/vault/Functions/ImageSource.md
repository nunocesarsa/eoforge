---
tags: [function, sources]
module: eoforge.data.sources
---
# ImageSource

> Base class for imagery sources: `prepare()` returns a local GeoTIFF path.

**Module:** [[eoforge.data.sources]] · **Source:** `eoforge/data/sources.py` · **Kind:** abstract class

```python
class ImageSource(ABC):
    def __init__(self, cfg: DataConfig, workdir: Path)
    def prepare(self) -> Path
```

## How it works

Subclass it, implement `prepare`, and decorate with [[register_source]]. `workdir` is the run folder, for caching downloads. `LocalGeoTIFF` is the working implementation; `STACSource` and `EarthEngineSource` are stubs.

## Used by

[[get_source]]

## See also

[[Roadmap]]
